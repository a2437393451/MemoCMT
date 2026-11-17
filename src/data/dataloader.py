import os
import pickle
import re
from typing import Any, Tuple, Union
import logging

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torchaudio

from configs.base import Config
from models.networks import MemoCMT
from tqdm.auto import tqdm


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        data_mode: str = "train.pkl",
        encoder_model: Union[MemoCMT, None] = None,
    ):
        super(BaseDataset, self).__init__()

        self.use_preencoded_embeddings = getattr(
            cfg, "use_preencoded_embeddings", False
        )
        self.data_root = (
            getattr(cfg, "preencoded_data_root", None)
            if self.use_preencoded_embeddings
            and getattr(cfg, "preencoded_data_root", None) is not None
            else cfg.data_root
        )

        with open(os.path.join(self.data_root, data_mode), "rb") as train_file:
            self.data_list = pickle.load(train_file)

        self.tokenizer = None
        if not self.use_preencoded_embeddings:
            if cfg.text_encoder_type == "bert":
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            else:
                raise NotImplementedError(
                    "Tokenizer {} is not implemented".format(cfg.text_encoder_type)
                )

        self.audio_max_length = cfg.audio_max_length
        self.text_max_length = cfg.text_max_length
        self.use_dynamic_padding = getattr(cfg, "use_dynamic_padding", False)
        if cfg.batch_size == 1:
            self.audio_max_length = None
            self.text_max_length = None

        self.audio_encoder_type = cfg.audio_encoder_type
        self.use_llm_explanations = getattr(cfg, "use_llm_explanations", False)
        self.llm_explanation_key = getattr(
            cfg, "llm_explanation_key", "llm_explanation"
        )
        self.llm_explanation_max_length = getattr(
            cfg, "llm_explanation_max_length", self.text_max_length
        )
        self.llm_explanation_fallback_to_text = getattr(
            cfg, "llm_explanation_fallback_to_text", False
        )

        self.encode_data = False
        self.list_encode_audio_data = []
        self.list_encode_text_data = []
        if self.use_preencoded_embeddings and encoder_model is not None:
            raise ValueError(
                "encoder_model preprocessing and preencoded dataset mode cannot be used together"
            )
        if encoder_model is not None:
            self._encode_data(encoder_model)
            self.encode_data = True

    def _encode_data(self, encoder):
        logging.info("Encoding data for training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.train()
        encoder.to(device)
        for index in tqdm(range(len(self.data_list))):
            sample = self.data_list[index]
            audio_path, text, _ = self._unpack_sample(sample)

            samples = self.__paudio__(audio_path)
            audio_embedding = (
                encoder.encode_audio(samples.unsqueeze(0).to(device))
                .squeeze(0)
                .detach()
                .cpu()
            )
            self.list_encode_audio_data.append(audio_embedding)

            input_ids = self.__ptext__(text)
            text_embedding = (
                encoder.encode_text(input_ids.unsqueeze(0).to(device))
                .squeeze(0)
                .detach()
                .cpu()
            )
            self.list_encode_text_data.append(text_embedding)

    def __getitem__(self, index: int):
        sample = self.data_list[index]
        if self.use_preencoded_embeddings:
            return self._get_preencoded_item(sample)

        audio_path, text, label = self._unpack_sample(sample)
        input_audio = (
            self.list_encode_audio_data[index]
            if self.encode_data
            else self.__paudio__(audio_path)
        )
        primary_text = self.__ptext__(text)
        input_text = self.list_encode_text_data[index] if self.encode_data else primary_text
        label = self.__plabel__(label)

        if not self.use_llm_explanations:
            return input_text, input_audio, label

        llm_explanation = self._get_llm_explanation(sample, text)
        input_llm_text = self.__ptext__(
            llm_explanation, max_length=self.llm_explanation_max_length
        )

        return input_text, input_audio, input_llm_text, label

    def _get_preencoded_item(self, sample):
        if not isinstance(sample, dict):
            raise ValueError("Preencoded dataset samples must be stored as dictionaries")

        text_embedding = torch.as_tensor(sample["text_embedding"], dtype=torch.float32)
        audio_embedding = torch.as_tensor(sample["audio_embedding"], dtype=torch.float32)
        text_mask = torch.as_tensor(sample["text_mask"], dtype=torch.bool)
        audio_mask = torch.as_tensor(sample["audio_mask"], dtype=torch.bool)
        label = self.__plabel__(sample["label"])

        llm_embedding = sample.get("llm_embedding")
        if llm_embedding is None or not self.use_llm_explanations:
            return text_embedding, audio_embedding, text_mask, audio_mask, label

        llm_embedding = torch.as_tensor(llm_embedding, dtype=torch.float32)
        llm_mask = torch.as_tensor(sample["llm_mask"], dtype=torch.bool)
        return (
            text_embedding,
            audio_embedding,
            llm_embedding,
            text_mask,
            audio_mask,
            llm_mask,
            label,
        )

    def _unpack_sample(
        self, sample: Union[Tuple[Any, ...], dict]
    ) -> Tuple[str, str, int]:
        if isinstance(sample, dict):
            return sample["audio_path"], sample["text"], sample["label"]
        if len(sample) < 3:
            raise ValueError("Each sample must contain at least audio_path, text and label")
        return sample[0], sample[1], sample[2]

    def _get_llm_explanation(
        self, sample: Union[Tuple[Any, ...], dict], text: str
    ) -> str:
        llm_explanation = ""
        if isinstance(sample, dict):
            llm_explanation = str(sample.get(self.llm_explanation_key, "") or "")
        elif len(sample) > 3 and isinstance(sample[3], str):
            llm_explanation = sample[3]

        if not llm_explanation and self.llm_explanation_fallback_to_text:
            llm_explanation = text
        return llm_explanation

    def __paudio__(self, file_path: int) -> torch.Tensor:
        samples, sr = sf.read(file_path, dtype="int16")
        samples = torch.from_numpy(samples.astype(np.float32))
        if self.use_dynamic_padding:
            if self.audio_max_length is not None:
                samples = samples[: self.audio_max_length]
            return torchaudio.functional.resample(samples, sr, 16000)

        if self.audio_max_length is not None and samples.shape[0] < self.audio_max_length:
            pad_size = self.audio_max_length - samples.shape[0]
            samples = torch.nn.functional.pad(samples, (0, pad_size))
        elif self.audio_max_length is not None:
            samples = samples[: self.audio_max_length]

        return torchaudio.functional.resample(samples, sr, 16000)

    def _text_preprocessing(self, text):
        text = re.sub("[\(\[].*?[\)\]]", "", text)
        text = re.sub(" +", " ", text).strip()
        try:
            text = " ".join(text.split())
        except Exception:
            text = "NULL"

        if not text.strip():
            text = "NULL"

        return text

    def __ptext__(
        self, text: str, max_length: Union[int, None] = None
    ) -> torch.Tensor:
        text = self._text_preprocessing(text)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        max_length = self.text_max_length if max_length is None else max_length
        if self.use_dynamic_padding:
            if max_length is not None:
                input_ids = input_ids[:max_length]
            return torch.from_numpy(np.asarray(input_ids))
        if max_length is not None and len(input_ids) < max_length:
            input_ids = np.pad(
                input_ids,
                (0, max_length - len(input_ids)),
                "constant",
                constant_values=self.tokenizer.pad_token_id,
            )
        elif max_length is not None:
            input_ids = input_ids[:max_length]
        return torch.from_numpy(np.asarray(input_ids))

    def __plabel__(self, label: int) -> torch.Tensor:
        return torch.tensor(label)

    def __len__(self):
        return len(self.data_list)


def _build_masked_collate_fn(cfg: Config):
    pad_token_id = 0

    def collate(batch):
        if len(batch[0]) != 4:
            raise ValueError("Masked collation requires the LLM explanation branch")

        texts, audios, llm_texts, labels = zip(*batch)
        text_lengths = torch.tensor([item.size(0) for item in texts])
        audio_lengths = torch.tensor([item.size(0) for item in audios])
        llm_lengths = torch.tensor([item.size(0) for item in llm_texts])

        input_text = pad_sequence(texts, batch_first=True, padding_value=pad_token_id)
        input_audio = pad_sequence(audios, batch_first=True, padding_value=0.0)
        input_llm_text = pad_sequence(
            llm_texts, batch_first=True, padding_value=pad_token_id
        )

        text_mask = (
            torch.arange(input_text.size(1)).unsqueeze(0) < text_lengths.unsqueeze(1)
        )
        audio_mask = (
            torch.arange(input_audio.size(1)).unsqueeze(0) < audio_lengths.unsqueeze(1)
        )
        llm_mask = (
            torch.arange(input_llm_text.size(1)).unsqueeze(0) < llm_lengths.unsqueeze(1)
        )

        return (
            input_text,
            input_audio,
            input_llm_text,
            text_mask,
            audio_mask,
            llm_mask,
            torch.stack(labels),
        )

    return collate


def _build_preencoded_collate_fn(cfg: Config):
    def collate(batch):
        if len(batch[0]) == 7:
            (
                text_embeddings,
                audio_embeddings,
                llm_embeddings,
                text_masks,
                audio_masks,
                llm_masks,
                labels,
            ) = zip(*batch)
            return (
                pad_sequence(text_embeddings, batch_first=True, padding_value=0.0),
                pad_sequence(audio_embeddings, batch_first=True, padding_value=0.0),
                pad_sequence(llm_embeddings, batch_first=True, padding_value=0.0),
                pad_sequence(text_masks, batch_first=True, padding_value=False),
                pad_sequence(audio_masks, batch_first=True, padding_value=False),
                pad_sequence(llm_masks, batch_first=True, padding_value=False),
                torch.stack(labels),
            )

        if len(batch[0]) == 5:
            text_embeddings, audio_embeddings, text_masks, audio_masks, labels = zip(*batch)
            return (
                pad_sequence(text_embeddings, batch_first=True, padding_value=0.0),
                pad_sequence(audio_embeddings, batch_first=True, padding_value=0.0),
                pad_sequence(text_masks, batch_first=True, padding_value=False),
                pad_sequence(audio_masks, batch_first=True, padding_value=False),
                torch.stack(labels),
            )

        raise ValueError("Unexpected preencoded batch format")

    return collate


def build_train_test_dataset(cfg: Config, encoder_model: Union[MemoCMT, None] = None):
    dataset_map = {
        "IEMOCAP": BaseDataset,
        "ESD": BaseDataset,
    }

    dataset = dataset_map.get(cfg.data_name, None)
    if dataset is None:
        raise NotImplementedError(
            "Dataset {} is not implemented, list of available datasets: {}".format(
                cfg.data_name, dataset_map.keys()
            )
        )
    if cfg.data_name in ["IEMOCAP_MSER", "MELD_MSER"]:
        return dataset(cfg)

    train_data = dataset(
        cfg,
        data_mode="train.pkl",
        encoder_model=encoder_model,
    )

    if encoder_model is not None:
        encoder_model.eval()
    test_set = cfg.data_valid if cfg.data_valid is not None else "test.pkl"
    test_data = dataset(
        cfg,
        data_mode=test_set,
        encoder_model=encoder_model,
    )

    collate_fn = None
    if getattr(cfg, "use_preencoded_embeddings", False):
        collate_fn = _build_preencoded_collate_fn(cfg)
    elif getattr(cfg, "use_attention_masks", False):
        collate_fn = _build_masked_collate_fn(cfg)

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return (train_dataloader, test_dataloader)

