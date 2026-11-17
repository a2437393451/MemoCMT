import torch
import torch.nn as nn

from configs.base import Config

from .modules import build_audio_encoder, build_text_encoder


class MemoCMT(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(MemoCMT, self).__init__()
        self.use_preencoded_embeddings = getattr(
            cfg, "use_preencoded_embeddings", False
        )
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        self.text_attention = nn.MultiheadAttention(
            embed_dim=cfg.text_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.dropout = nn.Dropout(cfg.dropout)
        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.fusion_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        self.fusion_head_output_type = cfg.fusion_head_output_type

    def _encode_audio(
        self, input_audio: torch.Tensor, audio_attention_mask: torch.Tensor = None
    ):
        if self.use_preencoded_embeddings:
            return input_audio, audio_attention_mask
        if len(input_audio.size()) != 2:
            batch_size, num_samples = input_audio.size(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(
                input_audio.view(-1, *input_audio.shape[2:])
            ).last_hidden_state
            audio_embeddings = audio_embeddings.mean(1)
            return (
                audio_embeddings.view(
                    batch_size, num_samples, *audio_embeddings.shape[1:]
                ),
                None,
            )
        if audio_attention_mask is None:
            return self.audio_encoder(input_audio), None
        return self.audio_encoder(
            input_audio,
            attention_mask=audio_attention_mask,
            return_attention_mask=True,
        )

    def _pool_sequence(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None
    ):
        if self.fusion_head_output_type == "cls":
            return embeddings[:, 0, :]
        if self.fusion_head_output_type == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
                return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return embeddings.mean(dim=1)
        if self.fusion_head_output_type == "max":
            if attention_mask is not None:
                masked = embeddings.masked_fill(
                    ~attention_mask.unsqueeze(-1), torch.finfo(embeddings.dtype).min
                )
                return masked.max(dim=1)[0]
            return embeddings.max(dim=1)[0]
        if self.fusion_head_output_type == "min":
            if attention_mask is not None:
                masked = embeddings.masked_fill(
                    ~attention_mask.unsqueeze(-1), torch.finfo(embeddings.dtype).max
                )
                return masked.min(dim=1)[0]
            return embeddings.min(dim=1)[0]
        raise ValueError("Invalid fusion head output type")

    def _build_fusion_mask(self, first_mask, second_mask, first_len, second_len):
        if first_mask is None and second_mask is None:
            return None
        if first_mask is None:
            first_mask = torch.ones(
                (second_mask.size(0), first_len),
                dtype=torch.bool,
                device=second_mask.device,
            )
        if second_mask is None:
            second_mask = torch.ones(
                (first_mask.size(0), second_len),
                dtype=torch.bool,
                device=first_mask.device,
            )
        return torch.cat((first_mask, second_mask), dim=1)

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
        text_attention_mask: torch.Tensor = None,
        audio_attention_mask: torch.Tensor = None,
    ):
        if self.use_preencoded_embeddings:
            text_embeddings = input_text
        else:
            text_embeddings = self.text_encoder(
                input_text, attention_mask=text_attention_mask
            ).last_hidden_state
        audio_embeddings, audio_feature_mask = self._encode_audio(
            input_audio, audio_attention_mask
        )

        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            key_padding_mask=(
                ~text_attention_mask.bool()
                if text_attention_mask is not None
                else None
            ),
            average_attn_weights=False,
        )
        text_norm = self.dropout(self.text_layer_norm(self.text_linear(text_attention)))

        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            key_padding_mask=(
                ~audio_feature_mask.bool() if audio_feature_mask is not None else None
            ),
            average_attn_weights=False,
        )
        audio_norm = self.dropout(self.audio_layer_norm(self.audio_linear(audio_attention)))

        fusion_norm = torch.cat((text_norm, audio_norm), 1)
        fusion_norm = self.dropout(fusion_norm)
        fusion_mask = self._build_fusion_mask(
            audio_feature_mask,
            text_attention_mask,
            text_norm.size(1),
            audio_norm.size(1),
        )
        cls_token_final_fusion_norm = self._pool_sequence(fusion_norm, fusion_mask)

        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [
                text_attn_output_weights,
                audio_attn_output_weights,
            ]

        return out, cls_token_final_fusion_norm, text_norm, audio_norm

    def encode_audio(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_attention_mask: bool = False,
    ):
        if self.use_preencoded_embeddings:
            if return_attention_mask:
                return audio, attention_mask
            return audio
        if return_attention_mask:
            return self.audio_encoder(
                audio,
                attention_mask=attention_mask,
                return_attention_mask=True,
            )
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        if self.use_preencoded_embeddings:
            return input_ids
        return self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state


class LLMReliabilityMemoCMT(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(LLMReliabilityMemoCMT, self).__init__()
        self.use_preencoded_embeddings = getattr(
            cfg, "use_preencoded_embeddings", False
        )
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        self.llm_text_encoder = build_text_encoder(cfg.llm_text_encoder_type)
        self.llm_text_encoder.to(device)
        for param in self.llm_text_encoder.parameters():
            param.requires_grad = cfg.llm_text_unfreeze

        self.text_attention = nn.MultiheadAttention(
            embed_dim=cfg.text_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.llm_linear = nn.Linear(cfg.llm_text_encoder_dim, cfg.fusion_dim)
        self.llm_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.use_reliability_gate = getattr(cfg, "use_reliability_gate", True)
        if self.use_reliability_gate:
            self.reliability_gate = nn.Sequential(
                nn.Linear(cfg.fusion_dim * 3, cfg.reliability_gate_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.reliability_gate_hidden_dim, 3),
            )
        self.use_uncertainty_weighting = getattr(
            cfg, "use_uncertainty_weighting", False
        )
        self.use_uncertainty_fusion_weighting = getattr(
            cfg, "use_uncertainty_fusion_weighting", False
        )
        if self.use_uncertainty_weighting:
            uncertainty_hidden_dim = getattr(
                cfg, "uncertainty_hidden_dim", max(1, cfg.fusion_dim // 4)
            )
            self.uncertainty_head = nn.Sequential(
                nn.Linear(cfg.fusion_dim, uncertainty_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(uncertainty_hidden_dim, 1),
                nn.Softplus(),
            )

        self.dropout = nn.Dropout(cfg.dropout)
        self.linear_layer_output = cfg.linear_layer_output
        previous_dim = cfg.fusion_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        self.fusion_head_output_type = cfg.fusion_head_output_type

    def _encode_audio(
        self, input_audio: torch.Tensor, audio_attention_mask: torch.Tensor = None
    ):
        if self.use_preencoded_embeddings:
            return input_audio, audio_attention_mask
        if len(input_audio.size()) != 2:
            batch_size, num_samples = input_audio.size(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(
                input_audio.view(-1, *input_audio.shape[2:])
            ).last_hidden_state
            audio_embeddings = audio_embeddings.mean(1)
            return (
                audio_embeddings.view(
                    batch_size, num_samples, *audio_embeddings.shape[1:]
                ),
                None,
            )
        if audio_attention_mask is None:
            return self.audio_encoder(input_audio), None
        return self.audio_encoder(
            input_audio,
            attention_mask=audio_attention_mask,
            return_attention_mask=True,
        )

    def _pool_sequence(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None
    ):
        if self.fusion_head_output_type == "cls":
            return embeddings[:, 0, :]
        if self.fusion_head_output_type == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
                return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return embeddings.mean(dim=1)
        if self.fusion_head_output_type == "max":
            if attention_mask is not None:
                masked = embeddings.masked_fill(
                    ~attention_mask.unsqueeze(-1), torch.finfo(embeddings.dtype).min
                )
                return masked.max(dim=1)[0]
            return embeddings.max(dim=1)[0]
        if self.fusion_head_output_type == "min":
            if attention_mask is not None:
                masked = embeddings.masked_fill(
                    ~attention_mask.unsqueeze(-1), torch.finfo(embeddings.dtype).max
                )
                return masked.min(dim=1)[0]
            return embeddings.min(dim=1)[0]
        raise ValueError("Invalid fusion head output type")

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        input_llm_text: torch.Tensor,
        output_attentions: bool = False,
        text_attention_mask: torch.Tensor = None,
        audio_attention_mask: torch.Tensor = None,
        llm_attention_mask: torch.Tensor = None,
    ):
        if self.use_preencoded_embeddings:
            text_embeddings = input_text
            audio_embeddings, audio_feature_mask = input_audio, audio_attention_mask
            llm_embeddings = input_llm_text
        else:
            text_embeddings = self.text_encoder(
                input_text, attention_mask=text_attention_mask
            ).last_hidden_state
            audio_embeddings, audio_feature_mask = self._encode_audio(
                input_audio, audio_attention_mask
            )
            llm_embeddings = self.llm_text_encoder(
                input_llm_text, attention_mask=llm_attention_mask
            ).last_hidden_state

        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            key_padding_mask=(
                ~text_attention_mask.bool()
                if text_attention_mask is not None
                else None
            ),
            average_attn_weights=False,
        )
        text_norm = self.dropout(self.text_layer_norm(self.text_linear(text_attention)))

        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            key_padding_mask=(
                ~audio_feature_mask.bool() if audio_feature_mask is not None else None
            ),
            average_attn_weights=False,
        )
        audio_norm = self.dropout(self.audio_layer_norm(self.audio_linear(audio_attention)))

        llm_norm = self.dropout(self.llm_layer_norm(self.llm_linear(llm_embeddings)))

        text_repr = self._pool_sequence(text_norm, audio_feature_mask)
        audio_repr = self._pool_sequence(audio_norm, text_attention_mask)
        llm_repr = self._pool_sequence(llm_norm, llm_attention_mask)

        gate_input = torch.cat((text_repr, audio_repr, llm_repr), dim=1)
        if self.use_reliability_gate:
            gate_weights = torch.softmax(self.reliability_gate(gate_input), dim=1)
        else:
            gate_weights = torch.full(
                (gate_input.size(0), 3),
                1.0 / 3.0,
                dtype=gate_input.dtype,
                device=gate_input.device,
            )
        uncertainty = reliability_weights = pair_reliability_weights = None
        if self.use_uncertainty_weighting:
            text_uncertainty = self.uncertainty_head(text_repr)
            audio_uncertainty = self.uncertainty_head(audio_repr)
            llm_uncertainty = self.uncertainty_head(llm_repr)
            uncertainty = torch.cat(
                (text_uncertainty, audio_uncertainty, llm_uncertainty), dim=1
            )
            reliability_weights = torch.softmax(torch.exp(-uncertainty), dim=1)
            pair_reliability_weights = torch.stack(
                (
                    reliability_weights[:, 0] * reliability_weights[:, 1],
                    reliability_weights[:, 0] * reliability_weights[:, 2],
                    reliability_weights[:, 1] * reliability_weights[:, 2],
                ),
                dim=1,
            )
            if self.use_uncertainty_fusion_weighting:
                gate_weights = gate_weights * reliability_weights
                gate_weights = gate_weights / gate_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
        fused_repr = (
            gate_weights[:, 0:1] * text_repr
            + gate_weights[:, 1:2] * audio_repr
            + gate_weights[:, 2:3] * llm_repr
        )

        x = self.dropout(fused_repr)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        if output_attentions:
            return [out, fused_repr], [
                text_attn_output_weights,
                audio_attn_output_weights,
                gate_weights,
            ]

        output = (out, fused_repr, text_norm, audio_norm, llm_norm, gate_weights)
        if any(
            mask is not None
            for mask in (
                text_attention_mask,
                audio_feature_mask,
                llm_attention_mask,
            )
        ):
            output = output + (
                audio_feature_mask,
                text_attention_mask,
                llm_attention_mask,
            )
        elif self.use_uncertainty_weighting:
            output = output + (None, None, None)
        if self.use_uncertainty_weighting:
            output = output + (
                uncertainty,
                reliability_weights,
                pair_reliability_weights,
            )
        return output

    def encode_audio(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_attention_mask: bool = False,
    ):
        if self.use_preencoded_embeddings:
            if return_attention_mask:
                return audio, attention_mask
            return audio
        if return_attention_mask:
            return self.audio_encoder(
                audio,
                attention_mask=attention_mask,
                return_attention_mask=True,
            )
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        if self.use_preencoded_embeddings:
            return input_ids
        return self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state

    def encode_llm_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        if self.use_preencoded_embeddings:
            return input_ids
        return self.llm_text_encoder(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state


class TextOnly(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(TextOnly, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.text_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = self.text_encoder(input_text).last_hidden_state
        fusion_norm = self.dropout(text_embeddings)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, cls_token_final_fusion_norm


class AudioOnly(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(AudioOnly, self).__init__()

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        self.linear_layer_output = cfg.linear_layer_output

        self.dropout = nn.Dropout(cfg.dropout)

        previous_dim = cfg.audio_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        if len(input_audio.size()) != 2:
            batch_size, num_samples = input_audio.size(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(
                input_audio.view(-1, *input_audio.shape[2:])
            ).last_hidden_state
            audio_embeddings = audio_embeddings.mean(1)
            audio_embeddings = audio_embeddings.view(
                batch_size, num_samples, *audio_embeddings.shape[1:]
            )
        else:
            audio_embeddings = self.audio_encoder(input_audio)

        fusion_norm = self.dropout(audio_embeddings)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, cls_token_final_fusion_norm


