"""Optional observed musical meter, separate from melody and IPA stress targets."""

import math

import numpy as np

METER_FEATURE_SCHEME = "observed_meter_phase_v1"
METER_FEATURE_DIM = 6


def encode_meter(lines):
    """Each melody may supply `meter`: one dict or null per note.

    Dicts contain numerator, denominator and beat_position (zero based, in
    denominator-note units within the bar). Missing entries contribute nothing;
    neither durations nor lyric stress are used to guess a time signature.
    """
    features, observed = [], []
    for line in lines:
        melody = line["melody"]
        count = len(melody["midi_pitches"])
        entries = melody.get("meter")
        if entries is None:
            entries = [None] * count
        if not isinstance(entries, (list, tuple)) or len(entries) != count:
            raise ValueError("melody.meter must contain one entry per note")
        for entry in entries:
            if entry is None:
                features.append([0.0] * METER_FEATURE_DIM)
                observed.append(False)
                continue
            if not isinstance(entry, dict):
                raise ValueError("Meter entries must be objects or null")
            numerator, denominator, beat = (
                entry.get(key) for key in ("numerator", "denominator", "beat_position")
            )
            if (
                any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 1
                    for v in (numerator, denominator)
                )
                or denominator & (denominator - 1)
                or isinstance(beat, bool)
                or not isinstance(beat, (int, float))
                or not math.isfinite(beat)
                or not 0 <= beat < numerator
            ):
                raise ValueError("Invalid observed meter numerator, denominator or beat_position")
            bar_phase, beat_phase = 2 * math.pi * beat / numerator, 2 * math.pi * (beat % 1)
            features.append(
                [
                    math.sin(bar_phase),
                    math.cos(bar_phase),
                    math.sin(beat_phase),
                    math.cos(beat_phase),
                    math.log2(numerator),
                    math.log2(denominator),
                ]
            )
            observed.append(True)
    return np.asarray(features, dtype=np.float32), np.asarray(observed, dtype=np.bool_)
