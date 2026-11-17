import argparse
import ast
import copy
import logging
import os
import pickle
import shutil
import sys

import torch
from tqdm.auto import tqdm

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

from configs.base import Config
from data.dataloader import BaseDataset
from models import networks
from utils.configs import get_options

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def apply_overrides(cfg, overrides):
    for override in overrides or []:
        if "=" not in override:
            raise ValueError("Overrides must use KEY=VALUE syntax")
        key, raw_value = override.split("=", 1)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        setattr(cfg, key, value)


def build_raw_dataset_cfg(cfg: Config) -> Config:
    dataset_cfg = copy.deepcopy(cfg)
    dataset_cfg.use_preencoded_embeddings = False
    dataset_cfg.preencoded_data_root = None
    return dataset_cfg


def infer_output_root(cfg: Config, output_root: str = None) -> str:
    if output_root is not None:
        return output_root
    return cfg.data_root.rstrip("/\\") + "_preencoded"


def encode_split(cfg: Config, split_name: str, output_root: str, device: torch.device):
    raw_cfg = build_raw_dataset_cfg(cfg)
    dataset = BaseDataset(raw_cfg, data_mode=split_name)
    network = getattr(networks, raw_cfg.model_type)(raw_cfg)
    network.to(device)
    network.eval()

    cached_samples = []
    for index in tqdm(range(len(dataset)), desc=f"Encoding {split_name}"):
        raw_sample = dataset.data_list[index]
        unpacked = dataset[index]
        audio_path, text, label_value = dataset._unpack_sample(raw_sample)

        if len(unpacked) == 4:
            input_text, input_audio, input_llm_text, label = unpacked
            llm_mask = torch.ones(input_llm_text.size(0), dtype=torch.bool)
        else:
            input_text, input_audio, label = unpacked
            input_llm_text = None
            llm_mask = None

        text_mask = torch.ones(input_text.size(0), dtype=torch.bool)
        audio_mask = torch.ones(input_audio.size(0), dtype=torch.bool)

        with torch.no_grad():
            text_embedding = network.encode_text(
                input_text.unsqueeze(0).to(device),
                attention_mask=text_mask.unsqueeze(0).to(device),
            ).squeeze(0).cpu()
            audio_embedding, audio_feature_mask = network.encode_audio(
                input_audio.unsqueeze(0).to(device),
                attention_mask=audio_mask.unsqueeze(0).to(device),
                return_attention_mask=True,
            )
            audio_embedding = audio_embedding.squeeze(0).cpu()
            audio_feature_mask = audio_feature_mask.squeeze(0).cpu()

            if input_llm_text is not None:
                llm_embedding = network.encode_llm_text(
                    input_llm_text.unsqueeze(0).to(device),
                    attention_mask=llm_mask.unsqueeze(0).to(device),
                ).squeeze(0).cpu()
            else:
                llm_embedding = None

        cached_sample = {
            "audio_path": audio_path,
            "text": text,
            "label": int(label.item() if torch.is_tensor(label) else label_value),
            "text_embedding": text_embedding,
            "audio_embedding": audio_embedding,
            "text_mask": text_mask.cpu(),
            "audio_mask": audio_feature_mask,
        }

        if llm_embedding is not None:
            cached_sample["llm_embedding"] = llm_embedding
            cached_sample["llm_mask"] = llm_mask.cpu()
            if isinstance(raw_sample, dict) and cfg.llm_explanation_key in raw_sample:
                cached_sample[cfg.llm_explanation_key] = raw_sample[cfg.llm_explanation_key]

        cached_samples.append(cached_sample)

    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, split_name)
    with open(output_path, "wb") as f:
        pickle.dump(cached_samples, f)
    logging.info("Saved preencoded split to: %s", output_path)


def copy_metadata_files(cfg: Config, output_root: str):
    os.makedirs(output_root, exist_ok=True)
    for file_name in ["classes.json"]:
        src_path = os.path.join(cfg.data_root, file_name)
        dst_path = os.path.join(output_root, file_name)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            logging.info("Copied metadata file to: %s", dst_path)


def main(args):
    cfg = get_options(args.config)
    apply_overrides(cfg, args.override)

    output_root = infer_output_root(cfg, args.output_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_candidates = args.splits or ["train.pkl", "val.pkl", "test.pkl"]
    splits = [
        split_name
        for split_name in split_candidates
        if os.path.exists(os.path.join(cfg.data_root, split_name))
    ]
    if not splits:
        raise FileNotFoundError(
            "No dataset splits found under {} from candidates {}".format(
                cfg.data_root, split_candidates
            )
        )

    logging.info("Using source data root: %s", cfg.data_root)
    logging.info("Writing preencoded dataset to: %s", output_root)
    copy_metadata_files(cfg, output_root)

    for split_name in splits:
        encode_split(cfg, split_name, output_root, device)

    logging.info("Done. Set use_preencoded_embeddings=True for training/evaluation.")
    logging.info("Suggested preencoded_data_root=%s", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Directory for cached embedding pickles",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Split files to encode, e.g. train.pkl val.pkl test.pkl",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value using KEY=VALUE; repeat as needed",
    )
    main(parser.parse_args())

