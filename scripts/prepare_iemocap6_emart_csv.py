"""Convert EMART's public IEMOCAP train/dev/test CSV files to URM-ER pickles."""

import argparse
import csv
import glob
import json
import os
import pickle
from collections import Counter


STRING_LABELS = {
    "happy": 0,
    "hap": 0,
    "sad": 1,
    "neutral": 2,
    "neu": 2,
    "angry": 3,
    "anger": 3,
    "ang": 3,
    "excited": 4,
    "exc": 4,
    "frustrated": 5,
    "fru": 5,
}


def normalize_label(value):
    value = str(value).strip()
    try:
        label = int(float(value))
    except ValueError:
        label = STRING_LABELS[value.lower()]
    if label not in range(6):
        raise ValueError("Expected a six-class label, got {!r}".format(value))
    return label


def find_audio(data_root, utterance_id):
    session = int(utterance_id[3:5])
    dialogue_id = utterance_id[:-5]
    path = os.path.abspath(
        os.path.join(
            data_root,
            "Session{}".format(session),
            "sentences",
            "wav",
            dialogue_id,
            utterance_id + ".wav",
        )
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def convert_csv(csv_path, data_root):
    samples = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Speaker", "Utterance", "Emotion"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "{} must contain columns {} (found {})".format(
                    csv_path, sorted(required), reader.fieldnames
                )
            )
        for row in reader:
            utterance_id = row["Speaker"].strip()
            samples.append(
                (
                    find_audio(data_root, utterance_id),
                    row["Utterance"].strip(),
                    normalize_label(row["Emotion"]),
                )
            )
    return samples


def load_transcript_index(data_root):
    transcripts = {}
    pattern = os.path.join(
        data_root, "Session*", "dialog", "transcriptions", "*.txt"
    )
    for transcript_path in glob.glob(pattern):
        with open(transcript_path, "r") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                left, text = line.split(":", 1)
                utterance_id = left.split()[0]
                transcripts[utterance_id] = text.strip()
    return transcripts


def convert_emart_json(split_json, data_root):
    with open(split_json, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    transcripts = load_transcript_index(data_root)
    split_aliases = {"train": "train", "dev": "val", "test": "test"}
    splits = {}
    for source_split, output_split in split_aliases.items():
        samples = []
        for item in manifest[source_split]:
            utterance_id = str(item[0])
            emotion = str(item[6])
            if utterance_id not in transcripts:
                raise KeyError("Transcript not found for {}".format(utterance_id))
            samples.append(
                (
                    find_audio(data_root, utterance_id),
                    transcripts[utterance_id],
                    normalize_label(emotion),
                )
            )
        splits[output_split] = samples
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--csv_dir",
        help="Directory containing EMART's IEMOCAP_train/dev/test.csv files.",
    )
    source_group.add_argument(
        "--split_json",
        help="EMART's public train_split_iemocap/iemocap6_fold1.json file.",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.split_json:
        splits = convert_emart_json(args.split_json, args.data_root)
        source_names = {"split_json": os.path.abspath(args.split_json)}
    else:
        source_names = {
            "train": "IEMOCAP_train.csv",
            "val": "IEMOCAP_dev.csv",
            "test": "IEMOCAP_test.csv",
        }
        splits = {
            split: convert_csv(os.path.join(args.csv_dir, filename), args.data_root)
            for split, filename in source_names.items()
        }

    os.makedirs(args.output_dir, exist_ok=True)
    for split, samples in splits.items():
        with open(os.path.join(args.output_dir, split + ".pkl"), "wb") as handle:
            pickle.dump(samples, handle)

    metadata = {
        "protocol": "Exact train/dev/test membership from the EMART public split",
        "source_files": source_names,
        "split_sizes": {name: len(samples) for name, samples in splits.items()},
        "class_counts": {
            name: dict(sorted(Counter(sample[2] for sample in samples).items()))
            for name, samples in splits.items()
        },
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "classes.json"), "w") as handle:
        json.dump(
            {
                "Happy": 0,
                "Sad": 1,
                "Neutral": 2,
                "Angry": 3,
                "Excited": 4,
                "Frustrated": 5,
            },
            handle,
            indent=2,
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
