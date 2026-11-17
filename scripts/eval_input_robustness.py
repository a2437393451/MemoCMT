

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.append(str(SRC_DIR))
sys.path.append(str(SCRIPT_DIR))

import torch
from tqdm.auto import tqdm

from configs.base import Config
from data.dataloader import build_train_test_dataset
from eval import (
    calculate_accuracy,
    calculate_f1_score,
    forward_batch,
    resolve_best_checkpoint,
)
from models import networks


CONDITIONS = (
    ("clean", None, None),
    ("audio_snr_20db", 20.0, None),
    ("audio_snr_10db", 10.0, None),
    ("audio_snr_0db", 0.0, None),
    ("text_mask_10pct", None, 0.10),
    ("text_mask_20pct", None, 0.20),
    ("text_mask_30pct", None, 0.30),
)


def _clone_batch(batch):
    return tuple(value.clone() if torch.is_tensor(value) else value for value in batch)


def _batch_masks(batch):
    if len(batch) == 7:
        return batch[3].bool(), batch[4].bool()
    if len(batch) == 5:
        return batch[2].bool(), batch[3].bool()
    return None, None


def add_noise_at_snr(audio, audio_mask, snr_db, base_seed, sample_index):
    """Add zero-mean Gaussian noise only to valid (non-padding) samples."""
    output = audio.clone()
    for row_index in range(output.size(0)):
        valid = (
            audio_mask[row_index]
            if audio_mask is not None
            else torch.ones_like(output[row_index], dtype=torch.bool)
        )
        if not valid.any():
            continue
        generator = torch.Generator(device=output.device)
        generator.manual_seed(base_seed + sample_index * 1009 + row_index)
        noise = torch.randn(
            output[row_index].shape,
            generator=generator,
            dtype=output.dtype,
            device=output.device,
        )
        signal_power = output[row_index][valid].pow(2).mean().clamp_min(1e-12)
        noise_power = noise[valid].pow(2).mean().clamp_min(1e-12)
        scale = torch.sqrt(
            signal_power
            / (noise_power * output.new_tensor(10.0 ** (snr_db / 10.0)))
        )
        output[row_index][valid] = output[row_index][valid] + scale * noise[valid]
    return output


def random_mask_text(
    input_ids,
    text_mask,
    tokenizer,
    mask_ratio,
    base_seed,
    sample_index,
):
    """Replace a fixed fraction of non-special valid tokens with [MASK]."""
    if tokenizer is None or tokenizer.mask_token_id is None:
        raise ValueError("Text masking requires a tokenizer with mask_token_id")
    output = input_ids.clone()
    special_ids = set(tokenizer.all_special_ids)
    for row_index in range(output.size(0)):
        valid = (
            text_mask[row_index]
            if text_mask is not None
            else torch.ones_like(output[row_index], dtype=torch.bool)
        )
        candidates = torch.tensor(
            [
                position
                for position, token_id in enumerate(output[row_index].tolist())
                if bool(valid[position]) and token_id not in special_ids
            ],
            dtype=torch.long,
            device=output.device,
        )
        if candidates.numel() == 0:
            continue
        count = max(1, int(round(candidates.numel() * mask_ratio)))
        count = min(count, candidates.numel())
        generator = torch.Generator(device=output.device)
        generator.manual_seed(base_seed + sample_index * 1009 + row_index)
        selected = candidates[
            torch.randperm(
                candidates.numel(), generator=generator, device=output.device
            )[:count]
        ]
        output[row_index, selected] = tokenizer.mask_token_id
    return output


def perturb_batch(batch, tokenizer, snr_db, mask_ratio, seed, sample_index):
    batch = list(_clone_batch(batch))
    text_mask, audio_mask = _batch_masks(batch)
    if snr_db is not None:
        batch[1] = add_noise_at_snr(
            batch[1], audio_mask, snr_db, seed, sample_index
        )
    if mask_ratio is not None:
        batch[0] = random_mask_text(
            batch[0], text_mask, tokenizer, mask_ratio, seed, sample_index
        )
    return tuple(batch)


def load_model_and_data(run_dir, test_set):
    cfg = Config()
    cfg.load(os.path.join(run_dir, "cfg.log"))
    cfg.data_valid = test_set
    checkpoint_path, all_state_dict = resolve_best_checkpoint(run_dir, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = getattr(networks, cfg.model_type)(cfg).to(device)
    weight = torch.load(checkpoint_path, map_location=device)
    if all_state_dict:
        weight = weight["state_dict_network"]
    network.load_state_dict(weight)
    network.eval()

    _, test_loader = build_train_test_dataset(cfg)
    tokenizer = getattr(test_loader.dataset, "tokenizer", None)
    return cfg, network, test_loader, tokenizer, device, checkpoint_path


def evaluate_condition(
    network,
    test_loader,
    tokenizer,
    device,
    condition,
    snr_db,
    mask_ratio,
    perturbation_seed,
):
    y_true, y_pred, predictions = [], [], []
    for sample_index, batch in enumerate(tqdm(test_loader, desc=condition)):
        perturbed = perturb_batch(
            batch,
            tokenizer,
            snr_db,
            mask_ratio,
            perturbation_seed,
            sample_index,
        )
        with torch.no_grad():
            logits, labels = forward_batch(network, perturbed, device)
            predicted = logits.argmax(dim=1)
        labels_cpu = labels.detach().cpu().tolist()
        predicted_cpu = predicted.detach().cpu().tolist()
        for true_label, pred_label in zip(labels_cpu, predicted_cpu):
            y_true.append(true_label)
            y_pred.append(pred_label)
            predictions.append(
                {
                    "condition": condition,
                    "sample_index": len(predictions),
                    "true_label": true_label,
                    "predicted_label": pred_label,
                }
            )

    bacc, acc = calculate_accuracy(y_true, y_pred)
    macro_f1, weighted_f1 = calculate_f1_score(y_true, y_pred)
    return {
        "bacc": 100.0 * bacc,
        "acc": 100.0 * acc,
        "macro_f1": 100.0 * macro_f1,
        "weighted_f1": 100.0 * weighted_f1,
    }, predictions


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_and_sample_std(values):
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sample_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        nargs="+",
        required=True,
        help="One or more run directories containing cfg.log and weights/",
    )
    parser.add_argument("--test-set", default="test.pkl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturbation-seed", type=int, default=2025)
    args = parser.parse_args()

    per_run_rows = []
    prediction_rows = []
    for run_index, run_dir in enumerate(args.checkpoint_path, start=1):
        cfg, network, loader, tokenizer, device, checkpoint = load_model_and_data(
            run_dir, args.test_set
        )
        clean_macro_f1 = None
        for condition, snr_db, mask_ratio in CONDITIONS:
            metrics, predictions = evaluate_condition(
                network,
                loader,
                tokenizer,
                device,
                condition,
                snr_db,
                mask_ratio,
                args.perturbation_seed,
            )
            if condition == "clean":
                clean_macro_f1 = metrics["macro_f1"]
            row = {
                "run": run_index,
                "dataset": cfg.data_name,
                "condition": condition,
                "snr_db": "" if snr_db is None else snr_db,
                "mask_ratio": "" if mask_ratio is None else mask_ratio,
                **{key: round(value, 4) for key, value in metrics.items()},
                "delta_macro_f1": round(metrics["macro_f1"] - clean_macro_f1, 4),
                "checkpoint": checkpoint,
            }
            per_run_rows.append(row)
            for prediction in predictions:
                prediction_rows.append(
                    {
                        "run": run_index,
                        "dataset": cfg.data_name,
                        **prediction,
                    }
                )

    summary_rows = []
    grouped = defaultdict(list)
    for row in per_run_rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    for (dataset, condition), rows in grouped.items():
        summary = {"dataset": dataset, "condition": condition, "n_runs": len(rows)}
        for metric in ("bacc", "acc", "macro_f1", "weighted_f1", "delta_macro_f1"):
            mean, sample_std = mean_and_sample_std([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = round(mean, 4)
            summary[f"{metric}_sample_std"] = round(sample_std, 4)
        summary_rows.append(summary)

    write_csv(
        args.output_dir / "robustness_per_run.csv",
        per_run_rows,
        list(per_run_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "robustness_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "robustness_predictions.csv",
        prediction_rows,
        list(prediction_rows[0].keys()),
    )
    print(args.output_dir / "robustness_summary.csv")


if __name__ == "__main__":
    main()
