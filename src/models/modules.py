import torch
import torch.nn as nn
import torchaudio
from transformers import (
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaModel,
    FocalNetConfig,
    FocalNetModel,
)

from configs.base import Config


def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, output_attentions=True
    )
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


class HuBertBase(nn.Module):
    def __init__(self, **kwargs):
        super(HuBertBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()

    def forward(self, x, attention_mask=None, return_attention_mask=False):
        lengths = None
        if attention_mask is not None:
            lengths = attention_mask.long().sum(dim=1)

        features, output_lengths = self.model(x, lengths)
        if not return_attention_mask:
            return features

        if output_lengths is None:
            feature_mask = torch.ones(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        else:
            positions = torch.arange(features.size(1), device=features.device)
            output_lengths = output_lengths.to(features.device)
            feature_mask = positions.unsqueeze(0) < output_lengths.unsqueeze(1)
        return features, feature_mask


def build_hubert_base_encoder(cfg: Config) -> nn.Module:
    """A function to build hubert encoder"""
    return HuBertBase()


def build_audio_encoder(cfg: Config) -> nn.Module:
    """A function to build audio encoder

    Args:
        cfg (Config): Config object

    Returns:
        nn.Module: Audio encoder
    """
    type = cfg.audio_encoder_type

    encoders = {
        "hubert_base": build_hubert_base_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type](cfg)


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder

    Args:
        type (str, optional): Type of text encoder. Defaults to "bert".

    Returns:
        torch.nn.Module: Text encoder
    """
    encoders = {
        "bert": build_bert_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()
