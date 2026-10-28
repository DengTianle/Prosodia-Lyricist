"""Validate standalone evaluation options before checkpoint I/O."""

import pytest

from prosodia_lyricist.bridge_eval import evaluate_bridge


@pytest.mark.parametrize("kwargs", [
    {"split": "train"}, {"limit": 0}, {"batch_size": 0}, {"num_workers": -1},
    {"precision": "bf16", "device": "cpu"},
])
def test_invalid_settings_fail_before_loading_checkpoint(kwargs):
    with pytest.raises(ValueError):
        evaluate_bridge("unused", "unused", **kwargs)
