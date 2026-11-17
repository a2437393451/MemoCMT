import argparse
import glob
import logging
import os
import re
import sys
from typing import Dict, List, Tuple

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import matplotlib.pyplot as plt
import numpy as np
import torch
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import networks
from sklearn.metrics import auc, roc_curve
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEFAULT_LABEL_NAMES = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def find_checkpoint_folders(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    if os.path.exists(os.path.join(path, "cfg.log")) and os.path.isdir(
        os.path.join(path, "weights")
    ):
        return [path]

    candidates = []
    for root, dirs, files in os.walk(path):
        if "cfg.log" in files and "weights" in dirs:
            candidates.append(root)
    return sorted(candidates)


def resolve_checkpoint_weight(
    ckpt_dir: str, latest: bool, cfg: Config = None
) -> Tuple[str, bool]:
    if latest:
        pt_files = glob.glob(os.path.join(ckpt_dir, "weights", "*.pt"))
        if pt_files:
            return sorted(pt_files)[-1], True

        pth_files = glob.glob(os.path.join(ckpt_dir, "weights", "*.pth"))
        if pth_files:
            return sorted(pth_files)[-1], False

    metrics = [getattr(cfg, "checkpoint_metric", "acc"), "acc"]
    for metric in dict.fromkeys(metrics):
        best_pt = os.path.join(
            ckpt_dir, "weights", "best_{}".format(metric), "checkpoint_0_0.pt"
        )
        if os.path.exists(best_pt):
            return best_pt, True

        best_pth = os.path.join(
            ckpt_dir, "weights", "best_{}".format(metric), "checkpoint_0.pth"
        )
        if os.path.exists(best_pth):
            return best_pth, False

    all_pt = glob.glob(os.path.join(ckpt_dir, "weights", "**", "*.pt"), recursive=True)
    if all_pt:
        return sorted(all_pt)[-1], True

    all_pth = glob.glob(os.path.join(ckpt_dir, "weights", "**", "*.pth"), recursive=True)
    if all_pth:
        return sorted(all_pth)[-1], False

    raise FileNotFoundError(f"No .pt or .pth checkpoint found under: {ckpt_dir}")


def evaluate_scores(cfg: Config, checkpoint_path: str, all_state_dict: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = getattr(networks, cfg.model_type)(cfg)
    network.to(device)

    _, test_ds = build_train_test_dataset(cfg)
    weight = torch.load(checkpoint_path, map_location=torch.device(device))
    if all_state_dict:
        weight = weight["state_dict_network"]

    network.load_state_dict(weight, strict=False)
    network.eval()
    network.to(device)

    y_true = []
    y_scores = []

    for batch in tqdm(test_ds):
        if len(batch) == 7:
            (
                input_ids,
                audio,
                input_llm_text,
                text_attention_mask,
                audio_attention_mask,
                llm_attention_mask,
                label,
            ) = batch
            model_kwargs = {
                "text_attention_mask": text_attention_mask.to(device),
                "audio_attention_mask": audio_attention_mask.to(device),
                "llm_attention_mask": llm_attention_mask.to(device),
            }
            input_llm_text = input_llm_text.to(device)
        elif len(batch) == 5:
            input_ids, audio, text_attention_mask, audio_attention_mask, label = batch
            input_llm_text = None
            model_kwargs = {
                "text_attention_mask": text_attention_mask.to(device),
                "audio_attention_mask": audio_attention_mask.to(device),
            }
        elif len(batch) == 4:
            input_ids, audio, input_llm_text, label = batch
            input_llm_text = input_llm_text.to(device)
            model_kwargs = {}
        else:
            input_ids, audio, label = batch
            input_llm_text = None
            model_kwargs = {}

        input_ids = input_ids.to(device)
        audio = audio.to(device)

        with torch.no_grad():
            if input_llm_text is not None:
                logits = network(
                    input_ids, audio, input_llm_text, **model_kwargs
                )[0]
            elif model_kwargs:
                logits = network(input_ids, audio, **model_kwargs)[0]
            else:
                logits = network(input_ids, audio)[0]
            probs = torch.softmax(logits, dim=1)

        y_true.append(int(label.detach().cpu().numpy()[0]))
        y_scores.append(probs.detach().cpu().numpy()[0])

    return np.asarray(y_true), np.asarray(y_scores)


def get_label_names(cfg: Config) -> List[str]:
    if getattr(cfg, "num_classes", None) is None:
        return DEFAULT_LABEL_NAMES

    if cfg.num_classes <= len(DEFAULT_LABEL_NAMES):
        return DEFAULT_LABEL_NAMES[: cfg.num_classes]

    return [f"Class {i}" for i in range(cfg.num_classes)]


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    label_names: List[str],
    title: str,
    output_path: str,
):
    classes = sorted(set(y_true.tolist()))
    plt.figure(figsize=(8, 5.5))

    for cls in classes:
        if cls >= y_scores.shape[1]:
            logging.warning(
                "Skipping class %s because score dimension is %s",
                cls,
                y_scores.shape[1],
            )
            continue

        y_true_binary = (y_true == cls).astype(int)
        if len(np.unique(y_true_binary)) < 2:
            logging.warning(
                "Skipping class %s because test set has only one label value", cls
            )
            continue

        fpr, tpr, _ = roc_curve(y_true_binary, y_scores[:, cls])
        roc_auc = auc(fpr, tpr)
        label = label_names[cls] if cls < len(label_names) else f"Class {cls}"
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "r--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()
    logging.info("Saved AUC/ROC curve to: %s", output_path)


def build_checkpoint_map(args) -> Dict[str, str]:
    ckpt_dirs = find_checkpoint_folders(args.checkpoint_path)
    if not ckpt_dirs:
        raise FileNotFoundError(
            f"No checkpoint folders with cfg.log and weights/ found under: {args.checkpoint_path}"
        )

    checkpoints = {}
    for ckpt_dir in ckpt_dirs:
        model_name = os.path.basename(os.path.dirname(ckpt_dir))
        time_name = os.path.basename(ckpt_dir)
        name = f"{model_name}_{time_name}"
        checkpoints[name] = ckpt_dir
    return checkpoints


def main(args):
    checkpoints = build_checkpoint_map(args)
    test_set = args.test_set if args.test_set is not None else "test.pkl"
    os.makedirs(args.output_dir, exist_ok=True)

    for name, ckpt_dir in checkpoints.items():
        logging.info("Evaluating checkpoint: %s", ckpt_dir)
        cfg_path = os.path.join(ckpt_dir, "cfg.log")
        cfg = Config()
        cfg.load(cfg_path)
        ckpt_path, all_state_dict = resolve_checkpoint_weight(
            ckpt_dir, args.latest, cfg
        )
        cfg.data_valid = test_set

        if args.data_root is not None:
            assert args.data_name is not None, "Changing dataset path requires --data_name"
            cfg.data_root = args.data_root
            cfg.data_name = args.data_name

        y_true, y_scores = evaluate_scores(
            cfg, ckpt_path, all_state_dict=all_state_dict
        )
        label_names = get_label_names(cfg)

        output_name = (
            f"AUC_ROC_{safe_name(name)}_{safe_name(cfg.data_name)}_{safe_name(test_set)}.png"
        )
        output_path = os.path.join(args.output_dir, output_name)
        title = f"AUC/ROC - {name} - {cfg.data_name} - {test_set}"
        plot_roc_curve(y_true, y_scores, label_names, title, output_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to a checkpoint folder or a parent folder containing checkpoint folders",
    )
    parser.add_argument(
        "-l",
        "--latest",
        action="store_true",
        help="Use the latest weight under weights/ instead of weights/best_acc",
    )
    parser.add_argument(
        "-t",
        "--test_set",
        type=str,
        default="test.pkl",
        help="Name of testing set, e.g. test.pkl",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override cfg.data_root",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        help="Override cfg.data_name when --data_root is used",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save AUC/ROC png files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)

