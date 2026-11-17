import logging
import os
import csv
from typing import Dict

import torch
from torch import Tensor
from sklearn.metrics import f1_score

from configs.base import Config
from models.networks import MemoCMT
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: MemoCMT,
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self._validation_targets = []
        self._validation_predictions = []

    def _prepare_batch(self, batch):
        if len(batch) == 7:
            (
                input_text,
                input_audio,
                input_llm_text,
                text_attention_mask,
                audio_attention_mask,
                llm_attention_mask,
                label,
            ) = batch
        elif len(batch) == 5:
            (
                input_text,
                input_audio,
                text_attention_mask,
                audio_attention_mask,
                label,
            ) = batch
            input_llm_text = None
            llm_attention_mask = None
        elif len(batch) == 4:
            input_text, input_audio, input_llm_text, label = batch
            text_attention_mask = audio_attention_mask = llm_attention_mask = None
        else:
            input_text, input_audio, label = batch
            input_llm_text = None
            text_attention_mask = audio_attention_mask = llm_attention_mask = None

        input_text = input_text.to(self.device)
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        if input_llm_text is not None:
            input_llm_text = input_llm_text.to(self.device)
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(self.device)
        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(self.device)
        if llm_attention_mask is not None:
            llm_attention_mask = llm_attention_mask.to(self.device)

        return (
            input_text,
            input_audio,
            input_llm_text,
            text_attention_mask,
            audio_attention_mask,
            llm_attention_mask,
            label,
        )

    def _forward_batch(self, prepared_batch):
        (
            input_text,
            input_audio,
            input_llm_text,
            text_attention_mask,
            audio_attention_mask,
            llm_attention_mask,
            _,
        ) = prepared_batch
        if input_llm_text is None:
            if text_attention_mask is None and audio_attention_mask is None:
                return self.network(input_text, input_audio)
            return self.network(
                input_text,
                input_audio,
                text_attention_mask=text_attention_mask,
                audio_attention_mask=audio_attention_mask,
            )
        if text_attention_mask is None:
            return self.network(input_text, input_audio, input_llm_text)
        return self.network(
            input_text,
            input_audio,
            input_llm_text,
            text_attention_mask=text_attention_mask,
            audio_attention_mask=audio_attention_mask,
            llm_attention_mask=llm_attention_mask,
        )

    def _loss_logs(self, loss):
        logs = {"loss": loss.detach().cpu().item()}
        if not getattr(self.cfg, "use_attention_masks", False):
            return logs
        components = getattr(self.criterion, "last_components", {})
        if components:
            logs["ce_loss"] = components["ce_loss"].cpu().item()
            logs["contrastive_loss"] = components["contrastive_loss"].cpu().item()
            if "uncertainty_loss" in components:
                logs["uncertainty_loss"] = components[
                    "uncertainty_loss"
                ].cpu().item()
        else:
            logs["ce_loss"] = logs["loss"]
            logs["contrastive_loss"] = 0.0
        return logs

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        prepared_batch = self._prepare_batch(batch)
        label = prepared_batch[-1]
        output = self._forward_batch(prepared_batch)
        loss = self.criterion(output, label)

        loss.backward()
        self.optimizer.step()

        _, preds = torch.max(output[0], 1)
        accuracy = torch.mean((preds == label).float())
        logs = self._loss_logs(loss)
        logs["acc"] = accuracy.detach().cpu().item()
        return logs

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        prepared_batch = self._prepare_batch(batch)
        label = prepared_batch[-1]
        with torch.no_grad():
            output = self._forward_batch(prepared_batch)
            loss = self.criterion(output, label)
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        self._validation_targets.extend(label.detach().cpu().tolist())
        self._validation_predictions.extend(preds.detach().cpu().tolist())
        logs = self._loss_logs(loss)
        logs["acc"] = accuracy.detach().cpu().item()
        return logs

    def reset_validation_metrics(self):
        self._validation_targets = []
        self._validation_predictions = []

    def finalize_validation_metrics(self):
        if (
            not getattr(self.cfg, "use_attention_masks", False)
            or not self._validation_targets
        ):
            return {}
        return {
            "macro_f1": f1_score(
                self._validation_targets,
                self._validation_predictions,
                average="macro",
            ),
            "weighted_f1": f1_score(
                self._validation_targets,
                self._validation_predictions,
                average="weighted",
            ),
        }

    def on_validation_end(self, logs, epoch, step):
        if not getattr(self.cfg, "use_attention_masks", False):
            return
        path = os.path.join(self.cfg.checkpoint_dir, "validation_metrics.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["epoch", "step"] + sorted(logs.keys())
            )
            if write_header:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "step": step, **logs})
