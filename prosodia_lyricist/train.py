"""Train the supervised melody-to-IPA bridge on prepared DALI songs."""

import argparse
import logging

import torch
from transformers import get_linear_schedule_with_warmup

from .config import load_config

logger = logging.getLogger(__name__)


def build_scheduler(optimizer, settings, total_steps):
    """The original literal -1 schedule is preserved as an explicit experiment choice."""
    mode = settings.get("schedule", "linear")
    if mode not in ("xai_original", "linear", "constant_after_warmup"):
        raise ValueError("Unknown schedule")
    warmup = settings.get("warmup_steps", int(total_steps * settings.get("warmup_ratio", 0)))
    if warmup < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if mode == "constant_after_warmup":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: step / warmup if step < warmup else 1.0
        )
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=-1 if mode == "xai_original" else total_steps,
    )


def train(config, *, output_dir=None, smoke_test=False):
    if config["model"].get("conditioning") != "bridge":
        raise ValueError("Only supervised bridge training is supported; use configs/bridge.yaml")
    from .bridge_train import train_bridge

    return train_bridge(config, output_dir=output_dir, smoke_test=smoke_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="New run directory (must not exist)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Two train/valid batches using the configured melody initialization",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger.info("Loading training config: %s", args.config)
    train(load_config(args.config), output_dir=args.output_dir, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
