import argparse
import glob
import json
import os
from typing import Dict, List, Tuple


def load_annotation_file(
    path: str, explanation_key: str
) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return [(str(text), str(explanation)) for text, explanation in data.items()]

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON object or a JSON list")

    annotations: List[Tuple[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item at index {index} must be an object")
        if "text" not in item:
            raise ValueError(f"{path}: item at index {index} is missing 'text'")
        if explanation_key not in item:
            raise ValueError(
                f"{path}: item at index {index} is missing '{explanation_key}'"
            )
        annotations.append((str(item["text"]), str(item[explanation_key])))
    return annotations


def collect_input_files(input_dir: str, pattern: str) -> List[str]:
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    return [
        path
        for path in files
        if os.path.isfile(path) and not path.endswith("_manifest.json")
    ]


def merge_annotations(
    file_paths: List[str], explanation_key: str, allow_overwrite: bool
) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}

    for path in file_paths:
        for text, explanation in load_annotation_file(path, explanation_key):
            if text in merged:
                existing = merged[text]
                if existing[explanation_key] != explanation and not allow_overwrite:
                    raise ValueError(
                        f"Conflicting explanations for text {text!r} found in "
                        f"{existing['source_file']} and {path}"
                    )
            merged[text] = {
                "text": text,
                explanation_key: explanation,
                "source_file": path,
            }

    return merged


def save_output(
    output_json: str,
    merged: Dict[str, Dict[str, str]],
    explanation_key: str,
    output_format: str,
) -> None:
    if output_format == "map":
        data = {
            text: item[explanation_key]
            for text, item in sorted(merged.items(), key=lambda pair: pair[0])
        }
    else:
        data = [
            {
                "text": item["text"],
                explanation_key: item[explanation_key],
            }
            for _, item in sorted(merged.items(), key=lambda pair: pair[0])
        ]

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing batch annotation JSON files",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path used to save merged annotations",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern used to select input files inside input_dir",
    )
    parser.add_argument(
        "--explanation_key",
        type=str,
        default="llm_explanation",
        help="Field name used for explanation text",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="list",
        choices=["list", "map"],
        help="Output format: list keeps [{'text', explanation}], map keeps {text: explanation}",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow later files to overwrite conflicting explanations for the same text",
    )
    args = parser.parse_args()

    input_files = collect_input_files(args.input_dir, args.pattern)
    if not input_files:
        raise ValueError(
            f"No input files found in {args.input_dir} matching pattern {args.pattern}"
        )

    merged = merge_annotations(
        input_files, args.explanation_key, args.allow_overwrite
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    save_output(
        args.output_json, merged, args.explanation_key, args.output_format
    )

    print(
        f"Merged {len(input_files)} files with {len(merged)} unique utterances into {args.output_json}"
    )


if __name__ == "__main__":
    main()
