

import argparse
import copy
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


LABEL_NAMES = {0: "anger", 1: "happiness", 2: "sadness", 3: "neutral"}
LEXICON = {
    0: ("anger", "angry"),
    1: ("happiness", "happy", "joyful", "joy"),
    2: ("sadness", "sad", "sorrowful", "sorrow"),
    3: ("neutral",),
}
EXPLANATION_KEY = "llm_explanation"


def compile_patterns():
    return {
        label: re.compile(
            r"(?<![A-Za-z])(?:" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")(?![A-Za-z])",
            flags=re.IGNORECASE,
        )
        for label, words in LEXICON.items()
    }


PATTERNS = compile_patterns()


def get_label(sample):
    return int(sample["label"] if isinstance(sample, dict) else sample[2])


def get_explanation(sample):
    if isinstance(sample, dict):
        return str(sample.get(EXPLANATION_KEY, "") or "")
    return str(sample[3] if len(sample) > 3 else "")


def set_explanation(sample, explanation):
    sample = copy.deepcopy(sample)
    if isinstance(sample, dict):
        sample[EXPLANATION_KEY] = explanation
        return sample
    values = list(sample)
    while len(values) < 4:
        values.append("")
    values[3] = explanation
    return tuple(values) if isinstance(sample, tuple) else values


def mentioned_labels(text):
    return {label for label, pattern in PATTERNS.items() if pattern.search(text)}


def mask_label_words(text):
    for pattern in PATTERNS.values():
        text = pattern.sub("[EMOTION]", text)
    return text


def deranged_indices(size, rng):
    indices = list(range(size))
    if size < 2:
        return indices
    for _ in range(1000):
        rng.shuffle(indices)
        if all(i != indices[i] for i in range(size)):
            return indices
    return indices[1:] + indices[:1]


def audit(samples):
    counts = Counter(total=len(samples))
    per_class = defaultdict(Counter)
    for sample in samples:
        true_label = get_label(sample)
        hits = mentioned_labels(get_explanation(sample))
        row = per_class[LABEL_NAMES.get(true_label, str(true_label))]
        row["total"] += 1
        if hits:
            counts["with_label_word"] += 1
            row["with_label_word"] += 1
        if len(hits) > 1:
            counts["ambiguous"] += 1
            row["ambiguous"] += 1
        if len(hits) == 1:
            counts["identifiable"] += 1
            row["identifiable"] += 1
            if true_label in hits:
                counts["agreement"] += 1
                row["agreement"] += 1

    total = max(counts["total"], 1)
    identifiable = max(counts["identifiable"], 1)
    return {
        **dict(counts),
        "coverage_rate": counts["with_label_word"] / total,
        "agreement_rate_among_identifiable": counts["agreement"] / identifiable,
        "per_class": {name: dict(value) for name, value in per_class.items()},
    }


def process_split(source, masked_target, shuffled_target, seed):
    with source.open("rb") as handle:
        samples = pickle.load(handle)
    explanations = [get_explanation(sample) for sample in samples]
    order = deranged_indices(len(samples), random.Random(seed))
    masked = [set_explanation(sample, mask_label_words(explanation)) for sample, explanation in zip(samples, explanations)]
    shuffled = [set_explanation(sample, explanations[order[i]]) for i, sample in enumerate(samples)]
    for target, value in ((masked_target, masked), (shuffled_target, shuffled)):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(value, handle)
    return audit(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--splits", nargs="+", default=["train.pkl", "val.pkl", "test.pkl"])
    args = parser.parse_args()

    report = {"seed": args.seed, "lexicon": {str(k): v for k, v in LEXICON.items()}, "splits": {}}
    for offset, split in enumerate(args.splits):
        source = args.input_dir / split
        if not source.exists():
            continue
        report["splits"][split] = process_split(
            source,
            args.output_dir / "label_word_masking" / split,
            args.output_dir / "shuffled_explanation" / split,
            args.seed + offset,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "label_shortcut_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
