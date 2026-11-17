import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data.dataloader import _build_masked_collate_fn
from models.losses import CrossEntropyContrastiveLoss, CrossEntropyLoss
from models import networks


def loss_config(mode="supervised"):
    return SimpleNamespace(
        contrastive_loss_weight=0.05,
        contrastive_temperature=0.1,
        contrastive_mode=mode,
        fusion_head_output_type="cls",
    )


def model_config():
    return SimpleNamespace(
        text_encoder_type="bert",
        text_encoder_dim=4,
        text_unfreeze=False,
        audio_encoder_type="hubert_base",
        audio_encoder_dim=4,
        audio_unfreeze=False,
        llm_text_encoder_type="bert",
        llm_text_encoder_dim=4,
        llm_text_unfreeze=False,
        num_attention_head=1,
        dropout=0.0,
        fusion_dim=4,
        reliability_gate_hidden_dim=4,
        linear_layer_output=[],
        num_classes=4,
        fusion_head_output_type="cls",
    )


class EncoderOutput:
    def __init__(self, hidden_state):
        self.last_hidden_state = hidden_state


class MockTextEncoder(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        values = input_ids.float().unsqueeze(-1)
        hidden = torch.cat((values, values + 1, values + 2, values + 3), dim=-1)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        return EncoderOutput(hidden)


class MockAudioEncoder(nn.Module):
    def forward(self, audio, attention_mask=None, return_attention_mask=False):
        values = audio.unsqueeze(-1)
        hidden = torch.cat((values, values + 1, values + 2, values + 3), dim=-1)
        if return_attention_mask:
            return hidden, attention_mask.bool()
        return hidden


class ContrastiveLossTests(unittest.TestCase):
    def test_supervised_mode_marks_same_class_off_diagonal_as_positive(self):
        criterion = CrossEntropyContrastiveLoss(loss_config("supervised"))
        captured_masks = []

        def capture(_, mask):
            captured_masks.append(mask.detach().clone())
            return torch.tensor(0.0)

        criterion._directional_alignment_loss = capture
        embeddings = torch.eye(3)
        criterion._modal_alignment_loss(
            embeddings, embeddings, torch.tensor([0, 0, 1])
        )

        self.assertTrue(captured_masks[0][0, 1])
        self.assertTrue(captured_masks[0][1, 0])
        self.assertFalse(captured_masks[0][0, 2])

    def test_loss_is_finite_and_has_gradients_for_edge_case_batches(self):
        for labels in (
            torch.tensor([0]),
            torch.tensor([1, 1, 1]),
            torch.tensor([0, 1, 2]),
        ):
            batch_size = labels.size(0)
            logits = torch.randn(batch_size, 4, requires_grad=True)
            representations = [
                torch.randn(batch_size, 2, 4, requires_grad=True)
                for _ in range(3)
            ]
            output = (
                logits,
                torch.randn(batch_size, 4),
                representations[0],
                representations[1],
                representations[2],
            )
            criterion = CrossEntropyContrastiveLoss(loss_config("supervised"))
            loss = criterion(output, labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)
            for representation in representations:
                self.assertIsNotNone(representation.grad)


class MaskAndCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.text_patch = patch.object(
            networks, "build_text_encoder", side_effect=lambda _: MockTextEncoder()
        )
        self.audio_patch = patch.object(
            networks, "build_audio_encoder", side_effect=lambda _: MockAudioEncoder()
        )
        self.text_patch.start()
        self.audio_patch.start()

    def tearDown(self):
        self.audio_patch.stop()
        self.text_patch.stop()

    def test_masked_prediction_is_invariant_to_extra_right_padding(self):
        model = networks.LLMReliabilityMemoCMT(model_config())
        model.eval()

        short = model(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0.2, 0.4, 0.6]]),
            torch.tensor([[4, 5]]),
            text_attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.bool),
            audio_attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.bool),
            llm_attention_mask=torch.tensor([[1, 1]], dtype=torch.bool),
        )[0]
        padded = model(
            torch.tensor([[1, 2, 3, 0, 0]]),
            torch.tensor([[0.2, 0.4, 0.6, 0.0, 0.0]]),
            torch.tensor([[4, 5, 0, 0]]),
            text_attention_mask=torch.tensor(
                [[1, 1, 1, 0, 0]], dtype=torch.bool
            ),
            audio_attention_mask=torch.tensor(
                [[1, 1, 1, 0, 0]], dtype=torch.bool
            ),
            llm_attention_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        )[0]
        torch.testing.assert_close(short, padded, rtol=1e-5, atol=1e-6)

    def test_v1_and_baseline_interfaces_remain_supported(self):
        cfg = model_config()
        criterion = CrossEntropyLoss(cfg)
        labels = torch.tensor([1, 2])

        v1_model = networks.LLMReliabilityMemoCMT(cfg)
        v1_output = v1_model(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            torch.tensor([[5, 6], [7, 8]]),
        )
        self.assertEqual(len(v1_output), 6)
        self.assertTrue(torch.isfinite(criterion(v1_output, labels)))

        baseline_model = networks.MemoCMT(cfg)
        baseline_output = baseline_model(
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        )
        self.assertEqual(len(baseline_output), 4)
        self.assertTrue(torch.isfinite(criterion(baseline_output, labels)))

    def test_dynamic_collate_returns_padding_masks(self):
        collate = _build_masked_collate_fn(SimpleNamespace())
        batch = [
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([0.1, 0.2]),
                torch.tensor([4, 5]),
                torch.tensor(0),
            ),
            (
                torch.tensor([6]),
                torch.tensor([0.3, 0.4, 0.5]),
                torch.tensor([7]),
                torch.tensor(1),
            ),
        ]
        text, audio, llm, text_mask, audio_mask, llm_mask, labels = collate(batch)
        self.assertEqual(tuple(text.shape), (2, 3))
        self.assertEqual(tuple(audio.shape), (2, 3))
        self.assertEqual(tuple(llm.shape), (2, 2))
        self.assertEqual(text_mask.tolist(), [[True, True, True], [True, False, False]])
        self.assertEqual(audio_mask.tolist(), [[True, True, False], [True, True, True]])
        self.assertEqual(llm_mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(labels.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
