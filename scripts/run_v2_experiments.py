import argparse
import csv
import os
import re
import statistics
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASETS = {
    "IEMOCAP": os.path.join(
        PROJECT_ROOT, "src", "configs", "hubert_llm_racmt_contrastive.py"
    ),
    "ESD": os.path.join(
        PROJECT_ROOT, "src", "configs", "hubert_llm_racmt_contrastive_esd.py"
    ),
}

CANDIDATES = {
    "ce_mask": {
        "loss_type": "CrossEntropyLoss",
        "use_contrastive_loss": False,
        "contrastive_mode": "instance",
        "contrastive_loss_weight": 0.0,
        "contrastive_temperature": 0.07,
    },
    "instance_w010_t007": {
        "loss_type": "CrossEntropyContrastiveLoss",
        "use_contrastive_loss": True,
        "contrastive_mode": "instance",
        "contrastive_loss_weight": 0.1,
        "contrastive_temperature": 0.07,
    },
    "supervised_w003_t010": {
        "loss_type": "CrossEntropyContrastiveLoss",
        "use_contrastive_loss": True,
        "contrastive_mode": "supervised",
        "contrastive_loss_weight": 0.03,
        "contrastive_temperature": 0.1,
    },
    "supervised_w005_t010": {
        "loss_type": "CrossEntropyContrastiveLoss",
        "use_contrastive_loss": True,
        "contrastive_mode": "supervised",
        "contrastive_loss_weight": 0.05,
        "contrastive_temperature": 0.1,
    },
}


def format_override(key, value):
    if isinstance(value, str):
        return "{}={}".format(key, value)
    return "{}={!r}".format(key, value)


def build_command(dataset, candidate, seed):
    settings = CANDIDATES[candidate]
    name = "LLMReliabilityMemoCMT_{}_v2_{}".format(dataset, candidate)
    overrides = {
        "name": name,
        "random_seed": seed,
        **settings,
    }
    command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "train.py"),
        "--config",
        DATASETS[dataset],
    ]
    for key, value in overrides.items():
        command.extend(["--override", format_override(key, value)])
    return command


def run_training(dataset, candidate, seed, dry_run=False):
    command = build_command(dataset, candidate, seed)
    print("\n$ {}".format(subprocess.list2cmdline(command)), flush=True)
    if dry_run:
        return None

    process = subprocess.Popen(
        command,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    checkpoint_dir = None
    for line in process.stdout:
        print(line, end="")
        match = re.search(r"CHECKPOINT_DIR=(.+)$", line.strip())
        if match:
            checkpoint_dir = match.group(1)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if checkpoint_dir is None:
        raise RuntimeError("Training finished without reporting CHECKPOINT_DIR")
    return checkpoint_dir


def best_validation_macro_f1(checkpoint_dir):
    metrics_path = os.path.join(checkpoint_dir, "validation_metrics.csv")
    with open(metrics_path, newline="") as csvfile:
        rows = list(csv.DictReader(csvfile))
    return 100.0 * max(float(row["macro_f1"]) for row in rows)


def write_summary(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "phase",
        "candidate",
        "dataset",
        "seed",
        "best_val_macro_f1",
        "checkpoint_dir",
        "test_bacc",
        "test_acc",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_final_evaluation(checkpoint_dir, output_dir):
    command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "eval.py"),
        "--checkpoint_path",
        checkpoint_dir,
        "--test_set",
        "test.pkl",
        "--output_dir",
        output_dir,
    ]
    print("\n$ {}".format(subprocess.list2cmdline(command)), flush=True)
    process = subprocess.Popen(
        command,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    csv_path = None
    for line in process.stdout:
        print(line, end="")
        match = re.search(r"Saved evaluation CSV to:\s*(.+)$", line.strip())
        if match:
            csv_path = match.group(1)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if csv_path is None:
        raise RuntimeError("Evaluation finished without reporting its CSV path")
    with open(csv_path, newline="") as csvfile:
        result = next(csv.DictReader(csvfile))
    return {
        "test_bacc": float(result["BACC"]),
        "test_acc": float(result["ACC"]),
        "test_macro_f1": float(result["MACRO_F1"]),
        "test_weighted_f1": float(result["WEIGHTED_F1"]),
    }


def write_aggregate(summary_path, records, candidate):
    stem, extension = os.path.splitext(summary_path)
    aggregate_path = stem + "_aggregate" + (extension or ".csv")
    metrics = [
        "best_val_macro_f1",
        "test_bacc",
        "test_acc",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    rows = []
    for dataset in DATASETS:
        selected = [
            row
            for row in records
            if row["candidate"] == candidate and row["dataset"] == dataset
        ]
        aggregate = {
            "candidate": candidate,
            "dataset": dataset,
            "seeds": ",".join(str(row["seed"]) for row in selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            aggregate[metric + "_mean"] = statistics.mean(values)
            aggregate[metric + "_std"] = statistics.stdev(values)
        rows.append(aggregate)

    fields = ["candidate", "dataset", "seeds"]
    for metric in metrics:
        fields.extend([metric + "_mean", metric + "_std"])
    with open(aggregate_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote aggregate results to {}".format(aggregate_path))


def execute_run(phase, dataset, candidate, seed, dry_run):
    checkpoint_dir = run_training(dataset, candidate, seed, dry_run=dry_run)
    if dry_run:
        return None
    return {
        "phase": phase,
        "candidate": candidate,
        "dataset": dataset,
        "seed": seed,
        "best_val_macro_f1": best_validation_macro_f1(checkpoint_dir),
        "checkpoint_dir": checkpoint_dir,
    }


def select_best_candidate(records):
    baseline = {
        row["dataset"]: row["best_val_macro_f1"]
        for row in records
        if row["candidate"] == "ce_mask"
    }
    scores = {}
    for candidate in CANDIDATES:
        if candidate == "ce_mask":
            continue
        gains = [
            row["best_val_macro_f1"] - baseline[row["dataset"]]
            for row in records
            if row["candidate"] == candidate
        ]
        scores[candidate] = sum(gains) / len(gains)
    best = max(scores, key=scores.get)
    print("Candidate mean gains: {}".format(scores))
    print("Selected candidate: {}".format(best))
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Run the 12-run revised V2 screening and confirmation matrix."
    )
    parser.add_argument(
        "--summary",
        default=os.path.join(PROJECT_ROOT, "v2_experiment_summary.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    args = parser.parse_args()

    records = []
    for candidate in CANDIDATES:
        for dataset in DATASETS:
            record = execute_run("screen", dataset, candidate, 0, args.dry_run)
            if record is not None:
                records.append(record)
                write_summary(args.summary, records)

    if args.dry_run or args.screen_only:
        return

    best_candidate = select_best_candidate(records)
    for seed in (1, 2):
        for dataset in DATASETS:
            record = execute_run(
                "confirmation", dataset, best_candidate, seed, args.dry_run
            )
            records.append(record)
            write_summary(args.summary, records)

    if not args.skip_final_test:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.summary)), "v2_final_test_results"
        )
        for record in records:
            if record["candidate"] != best_candidate:
                continue
            record.update(run_final_evaluation(record["checkpoint_dir"], output_dir))
            write_summary(args.summary, records)
        write_aggregate(args.summary, records, best_candidate)


if __name__ == "__main__":
    main()
