"""Validate optional repetition controls for XAI lyric generation."""

import math


def resolve_repetition_settings(model, no_repeat_ngram_size=None, repetition_penalty=None):
    config = model.bart.generation_config
    if no_repeat_ngram_size is None:
        no_repeat_ngram_size = config.no_repeat_ngram_size
    if repetition_penalty is None:
        repetition_penalty = config.repetition_penalty
    if (
        isinstance(no_repeat_ngram_size, bool)
        or not isinstance(no_repeat_ngram_size, int)
        or no_repeat_ngram_size < 0
    ):
        raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(repetition_penalty)
        or repetition_penalty <= 0
    ):
        raise ValueError("repetition_penalty must be positive and finite")
    return {
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
    }
