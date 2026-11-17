import glob
import json
import os
import time
from openai import OpenAI

SYSTEM_PATH = "/opt/MemoCMT-main/MemoCMT-main/prompts/llm_explanation_system.txt"
BATCH_DIR = "/root/autodl-tmp/MemoCMT/dataset/esd_llm_batches"
OUTPUT_DIR = "/root/autodl-tmp/MemoCMT/dataset/esd_llm_batches_results"
FAILED_DIR = "/root/autodl-tmp/MemoCMT/dataset/esd_llm_batches_failed"
MODEL_NAME = "deepseek-v4-flash"
SLEEP_SECONDS = 1.0


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
        return text[start:end + 1]
    return text


def repair_json_with_deepseek(client: OpenAI, broken_text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
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
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    with open(SYSTEM_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)

    batch_paths = sorted(glob.glob(os.path.join(BATCH_DIR, "llm_annotation_batch_*.txt")))
    if not batch_paths:
        raise ValueError(f"No batch files found in {BATCH_DIR}")

    for batch_path in batch_paths:
        output_name = os.path.basename(batch_path).replace(".txt", ".json")
        output_path = os.path.join(OUTPUT_DIR, output_name)
        failed_raw_path = os.path.join(FAILED_DIR, output_name.replace(".json", ".raw.txt"))

        if os.path.exists(output_path):
            print(f"Skip existing: {output_path}")
            continue

        with open(batch_path, "r", encoding="utf-8") as f:
            user_prompt = f.read()

        print(f"Processing: {batch_path}")

        response = client.chat.completions.create(
            model=MODEL_NAME,
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
            print(f"Repairing invalid JSON: {batch_path}")
            try:
                text = repair_json_with_deepseek(client, raw_text)
                json.loads(text)
            except Exception as e:
                with open(failed_raw_path, "w", encoding="utf-8") as f:
                    f.write(raw_text)
                print(f"Failed: {batch_path}")
                print(f"Saved raw response to: {failed_raw_path}")
                print(f"Error: {e}")
                time.sleep(SLEEP_SECONDS)
                continue

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved: {output_path}")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()