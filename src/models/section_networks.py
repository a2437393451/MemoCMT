
import torch
import torch.nn as nn

from .modules import build_text_encoder


class ExplanationOnly(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.encoder = build_text_encoder(cfg.llm_text_encoder_type)
        self.encoder.to(device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = cfg.llm_text_unfreeze
        self.output_type = cfg.fusion_head_output_type
        self.dropout = nn.Dropout(cfg.dropout)
        self.hidden = nn.ModuleList()
        previous_dim = cfg.llm_text_encoder_dim
        for dimension in cfg.linear_layer_output:
            self.hidden.append(nn.Linear(previous_dim, dimension))
            previous_dim = dimension
        self.classifier = nn.Linear(previous_dim, cfg.num_classes)

    def _pool(self, embeddings, mask):
        if self.output_type == "cls":
            return embeddings[:, 0]
        if self.output_type == "mean":
            if mask is None:
                return embeddings.mean(1)
            weights = mask.unsqueeze(-1).to(embeddings.dtype)
            return (embeddings * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        if self.output_type in {"max", "min"}:
            use_max = self.output_type == "max"
            if mask is not None:
                fill = torch.finfo(embeddings.dtype).min if use_max else torch.finfo(embeddings.dtype).max
                embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), fill)
            return embeddings.max(1)[0] if use_max else embeddings.min(1)[0]
        raise ValueError("fusion_head_output_type must be cls, mean, max, or min")

    def forward(self, input_text, input_audio, input_llm_text,
                output_attentions=False, text_attention_mask=None,
                audio_attention_mask=None, llm_attention_mask=None):
        embeddings = self.encoder(
            input_llm_text, attention_mask=llm_attention_mask
        ).last_hidden_state
        representation = self._pool(embeddings, llm_attention_mask)
        value = self.dropout(representation)
        for layer in self.hidden:
            value = nn.functional.leaky_relu(layer(value))
        return self.classifier(self.dropout(value)), representation
