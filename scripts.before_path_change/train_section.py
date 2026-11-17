

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.append(str(SRC_DIR))
sys.path.append(str(SCRIPT_DIR))

import train
from models import networks
from models.section_4_8_networks import ExplanationOnly
from utils.configs import get_options


def disable_three_branch_losses(cfg):
    cfg.loss_type = "CrossEntropyLoss"
    cfg.use_contrastive_loss = False
    cfg.use_uncertainty_weighting = False
    cfg.use_uncertainty_fusion_weighting = False
    cfg.use_uncertainty_contrastive_weighting = False
    cfg.uncertainty_loss_weight = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Full-model V3 config")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["label_word_masking", "shuffled_explanation", "no_explanation", "explanation_only"],
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = get_options(args.config)
    cfg.data_root = args.data_root
    cfg.checkpoint_dir = args.checkpoint_root
    cfg.random_seed = args.seed
    cfg.name = f"section_4_8_{args.variant}"

    if args.variant == "no_explanation":
        cfg.model_type = "MemoCMT"
        cfg.use_llm_explanations = False
        disable_three_branch_losses(cfg)
        # Fixed-length collation keeps the existing training pipeline unchanged.
        cfg.use_attention_masks = False
        cfg.use_dynamic_padding = False
    elif args.variant == "explanation_only":
        networks.ExplanationOnly = ExplanationOnly
        cfg.model_type = "ExplanationOnly"
        cfg.use_llm_explanations = True
        disable_three_branch_losses(cfg)

    train.apply_overrides(cfg, args.override)
    train.main(cfg)


if __name__ == "__main__":
    main()
