"""Runtime choices shared by training and inference."""

import logging
import random
import sys
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch


@contextmanager
def log_stage(description, *args):
    """Expose slow startup I/O and CPU high-water memory in batch-job logs."""
    logger = logging.getLogger(__name__)
    logger.info(description + " ...", *args)
    started = perf_counter()
    yield
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux (including Slurm nodes) reports KiB.
        gib = peak / (1024**3 if sys.platform == "darwin" else 1024**2)
        memory = f"; process peak CPU RSS {gib:.2f} GiB"
    except ImportError:
        memory = ""
    logger.info(description + " finished in %.1fs%s", *args, perf_counter() - started, memory)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name="auto"):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(name)
