import argparse
import json
import math
import os
from typing import Dict, List


def load_annotation_template(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("annotation template must be a JSON list")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"template item at index {index} must be an object")
        if "text" not in item:
            raise ValueError(f"template item at index {index} is missing 'text'")

    return data


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_list(data: List[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def render_prompt(prompt_template: str, batch_data: List[Dict[str, str]]) -> str:
    utterance_json = json.dumps(batch_data, ensure_ascii=False, indent=2)
    return prompt_template.replace("{{UTTERANCE_JSON}}", utterance_json)


def write_json(path: str, data: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to annotation template JSON exported by export_annotation_template.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory used to save batch JSON and prompt files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of utterances per batch",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=os.path.join("prompts", "llm_explanation_user_template.txt"),
        help="Prompt template path containing {{UTTERANCE_JSON}} placeholder",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="llm_annotation_batch",
        help="Output file prefix",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    template_data = load_annotation_template(args.input_json)
    prompt_template = load_prompt_template(args.prompt_template)
    batches = chunk_list(template_data, args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)

    manifest = {
        "input_json": os.path.abspath(args.input_json),
        "prompt_template": os.path.abspath(args.prompt_template),
        "batch_size": args.batch_size,
        "total_items": len(template_data),
        "total_batches": len(batches),
        "batches": [],
    }

    batch_digits = max(3, len(str(len(batches))))
    for batch_index, batch_data in enumerate(batches, start=1):
        batch_id = str(batch_index).zfill(batch_digits)
        batch_json_name = f"{args.prefix}_{batch_id}.json"
        batch_prompt_name = f"{args.prefix}_{batch_id}.txt"
        batch_json_path = os.path.join(args.output_dir, batch_json_name)
        batch_prompt_path = os.path.join(args.output_dir, batch_prompt_name)

        write_json(batch_json_path, batch_data)
        write_text(batch_prompt_path, render_prompt(prompt_template, batch_data))

        manifest["batches"].append(
            {
                "batch_index": batch_index,
                "num_items": len(batch_data),
                "json_path": os.path.abspath(batch_json_path),
                "prompt_path": os.path.abspath(batch_prompt_path),
            }
        )

    manifest_path = os.path.join(args.output_dir, f"{args.prefix}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(
        f"Saved {len(template_data)} items into {len(batches)} batches under {args.output_dir}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
