import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CELoss
from configs.base import Config


class CrossEntropyLoss(CELoss):
    """Rewrite CrossEntropyLoss to support init with kwargs"""

    def __init__(self, cfg: Config, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = input[0]
        return super().forward(out, target)


class CrossEntropyContrastiveLoss(CELoss):
    """Cross entropy plus sample-level cross-modal contrastive alignment."""

    def __init__(self, cfg: Config, **kwargs):
        super(CrossEntropyContrastiveLoss, self).__init__(**kwargs)
        self.cfg = cfg
        self.contrastive_loss_weight = cfg.contrastive_loss_weight
        self.temperature = cfg.contrastive_temperature
        self.contrastive_mode = getattr(cfg, "contrastive_mode", "instance")
        self.use_uncertainty_contrastive_weighting = getattr(
            cfg, "use_uncertainty_contrastive_weighting", False
        )
        self.uncertainty_loss_weight = getattr(cfg, "uncertainty_loss_weight", 0.0)
        if self.contrastive_mode not in {"instance", "supervised"}:
            raise ValueError("contrastive_mode must be 'instance' or 'supervised'")
        self.last_components = {}

    def _pool_sequence(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        output_type = getattr(self.cfg, "fusion_head_output_type", "cls")
        if output_type == "cls":
            return embeddings[:, 0, :]
        if output_type == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
                return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return embeddings.mean(dim=1)
        if output_type == "max":
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(
                    ~attention_mask.unsqueeze(-1), torch.finfo(embeddings.dtype).min
                )
            return embeddings.max(dim=1)[0]
        if output_type == "min":
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(
                    ~attention_mask.unsqueeze(-1), torch.finfo(embeddings.dtype).max
                )
            return embeddings.min(dim=1)[0]
        raise ValueError("Invalid fusion head output type")

    def _directional_alignment_loss(
        self, logits: torch.Tensor, positive_mask: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        positive_mask = positive_mask.to(log_probs.dtype)
        positive_count = positive_mask.sum(dim=1).clamp_min(1.0)
        return -((log_probs * positive_mask).sum(dim=1) / positive_count).mean()

    def _modal_alignment_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        logits = torch.matmul(anchor, positive.transpose(0, 1)) / self.temperature
        if self.contrastive_mode == "supervised":
            positive_mask = target.unsqueeze(0) == target.unsqueeze(1)
        else:
            positive_mask = torch.eye(
                logits.size(0), dtype=torch.bool, device=logits.device
            )
        return 0.5 * (
            self._directional_alignment_loss(logits, positive_mask)
            + self._directional_alignment_loss(
                logits.transpose(0, 1), positive_mask.transpose(0, 1)
            )
        )

    def _contrastive_loss(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if len(input) < 5:
            raise ValueError(
                "CrossEntropyContrastiveLoss requires model outputs: "
                "out, fused_repr, text_norm, audio_norm, llm_norm"
            )

        text_mask = input[6] if len(input) >= 9 else None
        audio_mask = input[7] if len(input) >= 9 else None
        llm_mask = input[8] if len(input) >= 9 else None

        text_repr = self._pool_sequence(input[2], text_mask)
        audio_repr = self._pool_sequence(input[3], audio_mask)
        llm_repr = self._pool_sequence(input[4], llm_mask)

        text_audio_loss = self._modal_alignment_loss(text_repr, audio_repr, target)
        text_llm_loss = self._modal_alignment_loss(text_repr, llm_repr, target)
        audio_llm_loss = self._modal_alignment_loss(audio_repr, llm_repr, target)
        losses = torch.stack((text_audio_loss, text_llm_loss, audio_llm_loss))
        if self.use_uncertainty_contrastive_weighting and len(input) >= 12:
            pair_reliability_weights = input[11]
            if pair_reliability_weights is not None:
                weights = pair_reliability_weights.mean(dim=0).to(losses.dtype)
                return (losses * weights).sum() / weights.sum().clamp_min(1e-8)
        return losses.mean()

    def _uncertainty_loss(self, input: torch.Tensor) -> torch.Tensor:
        if self.uncertainty_loss_weight <= 0.0 or len(input) < 10:
            return input[0].new_tensor(0.0)
        uncertainty = input[9]
        if uncertainty is None:
            return input[0].new_tensor(0.0)
        return uncertainty.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = input[0]
        ce_loss = super().forward(out, target)
        contrastive_loss = self._contrastive_loss(input, target)
        uncertainty_loss = self._uncertainty_loss(input)
        total_loss = (
            ce_loss
            + self.contrastive_loss_weight * contrastive_loss
            + self.uncertainty_loss_weight * uncertainty_loss
        )
        self.last_components = {
            "total_loss": total_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
        }
        return total_loss
