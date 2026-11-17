import argparse
import glob
import json
import os
import time

from openai import OpenAI


DEFAULT_MODEL_NAME = "deepseek-v4-flash"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_array(text: str) -> str:
    text = strip_code_fence(text)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def repair_json_with_deepseek(client, broken_text: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert the user's content into valid JSON only. "
                    "Preserve the top-level list format and the fields "
                    "'text' and 'llm_explanation'. Do not add markdown."
                ),
            },
            {"role": "user", "content": broken_text},
        ],
        temperature=0.0,
        top_p=0.9,
    )
    return extract_json_array(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--failed_dir", required=True)
    parser.add_argument(
        "--system_prompt",
        default=os.path.join("prompts", "llm_explanation_system.txt"),
    )
    parser.add_argument("--prefix", default="iemocap6_missing_batch")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    with open(args.system_prompt, "r", encoding="utf-8") as handle:
        system_prompt = handle.read()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.failed_dir, exist_ok=True)
    batch_paths = sorted(
        glob.glob(os.path.join(args.batch_dir, args.prefix + "_*.txt"))
    )
    if not batch_paths:
        raise ValueError("No batch files found in {}".format(args.batch_dir))

    for batch_path in batch_paths:
        output_name = os.path.basename(batch_path).replace(".txt", ".json")
        output_path = os.path.join(args.output_dir, output_name)
        failed_path = os.path.join(
            args.failed_dir, output_name.replace(".json", ".raw.txt")
        )
        if os.path.exists(output_path):
            print("Skip existing: {}".format(output_path))
            continue

        with open(batch_path, "r", encoding="utf-8") as handle:
            user_prompt = handle.read()
        print("Processing: {}".format(batch_path))
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            top_p=0.9,
        )
        raw_text = response.choices[0].message.content
        text = extract_json_array(raw_text)
        try:
            json.loads(text)
        except json.JSONDecodeError:
            print("Repairing invalid JSON: {}".format(batch_path))
            try:
                text = repair_json_with_deepseek(client, raw_text, args.model)
                json.loads(text)
            except Exception as error:
                with open(failed_path, "w", encoding="utf-8") as handle:
                    handle.write(raw_text)
                print("Failed: {}".format(batch_path))
                print("Saved raw response to: {}".format(failed_path))
                print("Error: {}".format(error))
                time.sleep(args.sleep_seconds)
                continue

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("Saved: {}".format(output_path))
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
