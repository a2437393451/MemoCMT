import argparse
import json
import os
import pickle
from typing import Dict, Iterable, List, Tuple, Union


SampleType = Union[Tuple[str, str, int], Dict[str, object]]


def load_pickle(path: str) -> List[SampleType]:
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path: str, data: List[Dict[str, object]]) -> None:
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_explanation_map(annotation_json: str, explanation_key: str) -> Dict[str, str]:
    with open(annotation_json, "r", encoding="utf-8") as file:
        raw_annotations = json.load(file)

    if isinstance(raw_annotations, dict):
        return {str(key): str(value) for key, value in raw_annotations.items()}

    if isinstance(raw_annotations, list):
        explanation_map: Dict[str, str] = {}
        for index, item in enumerate(raw_annotations):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Annotation item at index {index} must be an object when using list format"
                )
            if "text" not in item:
                raise ValueError(
                    f"Annotation item at index {index} is missing required field 'text'"
                )
            if explanation_key not in item:
                raise ValueError(
                    f"Annotation item at index {index} is missing required field '{explanation_key}'"
                )
            explanation_map[str(item["text"])] = str(item[explanation_key])
        return explanation_map

    raise ValueError(
        "annotation_json must be either a JSON object mapping text to explanation or a list of objects"
    )


def normalize_samples(
    samples: Iterable[SampleType],
    explanation_map: Dict[str, str],
    explanation_key: str,
) -> List[Dict[str, object]]:
    normalized_samples: List[Dict[str, object]] = []
    for sample in samples:
        if isinstance(sample, dict):
            audio_path = str(sample["audio_path"])
            text = str(sample["text"])
            label = int(sample["label"])
            sample_data = dict(sample)
        else:
            audio_path, text, label = sample[:3]
            sample_data = {
                "audio_path": audio_path,
                "text": text,
                "label": label,
            }

        if explanation_key not in sample_data:
            sample_data[explanation_key] = explanation_map.get(text, "")
        normalized_samples.append(sample_data)
    return normalized_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing train.pkl/val.pkl/test.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save annotated pickles",
    )
    parser.add_argument(
        "--annotation_json",
        type=str,
        required=True,
        help="JSON file mapping transcript text to LLM explanations",
    )
    parser.add_argument(
        "--explanation_key",
        type=str,
        default="llm_explanation",
        help="Field name used to store explanation text",
    )
    args = parser.parse_args()

    explanation_map = load_explanation_map(
        args.annotation_json, args.explanation_key
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name in ("train.pkl", "val.pkl", "test.pkl"):
        input_path = os.path.join(args.input_dir, split_name)
        output_path = os.path.join(args.output_dir, split_name)
        samples = load_pickle(input_path)
        annotated_samples = normalize_samples(
            samples, explanation_map, args.explanation_key
        )
        save_pickle(output_path, annotated_samples)
        print(f"Saved {split_name} with {len(annotated_samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
