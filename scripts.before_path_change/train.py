import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import ast
import datetime
import random

import numpy as np
import torch
from torch import optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks, optims
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def main(cfg: Config):
    set_random_seed(getattr(cfg, "random_seed", 0))
    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)
    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(cfg.model_type))

    logging.info("Initializing checkpoint directory and dataset...")
    # Preapre the checkpoint directory
    run_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if getattr(cfg, "use_attention_masks", False):
        run_dir_name = "seed{}_{}".format(
            getattr(cfg, "random_seed", 0), run_dir_name
        )
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(cfg.checkpoint_dir),
        cfg.name,
        run_dir_name,
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    cfg.save(cfg)

    try:
        criterion = getattr(losses, cfg.loss_type)(cfg)
        criterion.to(device)
    except AttributeError:
        raise NotImplementedError("Loss {} is not implemented".format(cfg.loss_type))

    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError("Trainer {} is not implemented".format(cfg.trainer))

    train_ds, test_ds = build_train_test_dataset(cfg)
    logging.info("Initializing trainer...")

    logging.info("Start training...")

    optimizer = optims.get_optim(cfg, network)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
        monitored_metrics=(
            [cfg.checkpoint_metric]
            if getattr(cfg, "use_attention_masks", False)
            else None
        ),
    )

    if cfg.resume:
        trainer.load_all_states(cfg.resume_path)

    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback])
    logging.info("CHECKPOINT_DIR=%s", checkpoint_dir)
    return checkpoint_dir


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value using KEY=VALUE; repeat as needed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = get_options(args.config)
    apply_overrides(cfg, args.override)
    if cfg.resume and cfg.cfg_path is not None:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.cfg_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    main(cfg)
