import argparse
import json
import os
import pickle
from typing import Dict, Iterable, List, Set, Tuple, Union


SampleType = Union[Tuple[str, str, int], Dict[str, object]]


def load_pickle(path: str) -> List[SampleType]:
    with open(path, "rb") as file:
        return pickle.load(file)


def extract_text(sample: SampleType) -> str:
    if isinstance(sample, dict):
        return str(sample["text"])
    return str(sample[1])


def collect_unique_texts(input_dir: str) -> Set[str]:
    unique_texts: Set[str] = set()
    for split_name in ("train.pkl", "val.pkl", "test.pkl"):
        split_path = os.path.join(input_dir, split_name)
        samples = load_pickle(split_path)
        for sample in samples:
            text = extract_text(sample).strip()
            if text:
                unique_texts.add(text)
    return unique_texts


def build_template(texts: Iterable[str], explanation_key: str) -> List[Dict[str, str]]:
    template: List[Dict[str, str]] = []
    for text in sorted(texts):
        template.append(
            {
                "text": text,
                explanation_key: "",
            }
        )
    return template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing train.pkl/val.pkl/test.pkl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save the annotation template JSON",
    )
    parser.add_argument(
        "--explanation_key",
        type=str,
        default="llm_explanation",
        help="Field name used for explanation text",
    )
    args = parser.parse_args()

    texts = collect_unique_texts(args.input_dir)
    template = build_template(texts, args.explanation_key)

    with open(args.output_json, "w", encoding="utf-8") as file:
        json.dump(template, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(template)} unique texts to {args.output_json}")


if __name__ == "__main__":
    main()
