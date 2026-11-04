"""Versioned, note-centric musical inputs. No lyric/alignment fields are read here."""

import hashlib
import json
import math
from bisect import bisect_right
from fractions import Fraction

import numpy as np

REPRESENTATION = "prosody_octuple_v1"
GRID_POLICY = "explicit_meter_continuous_bars_v1"
FIELDS = ("bar", "position", "pitch", "interval", "duration", "gap", "meter", "tempo")
PAD, UNK, MASK, START = range(4)
OFFSET = 4
ROLE_START, ROLE_CONTINUE = range(2)


def codec_config(config=None):
    result = {"resolution": 96, "timing_residuals": True}
    if config is not None:
        if not isinstance(config, dict) or config.keys() - result.keys():
            raise ValueError("Octuple options are resolution and timing_residuals")
        result.update(config)
    if type(result["resolution"]) is not int or result["resolution"] not in (48, 96):
        raise ValueError("Octuple resolution must be 48 or 96")
    if type(result["timing_residuals"]) is not bool:
        raise ValueError("timing_residuals must be boolean")
    return result


def time_centers(resolution=96):
    return np.concatenate(
        (np.arange(4 * resolution + 1) / resolution, 4 * 2.0 ** (np.arange(1, 385) / 48))
    )


def codec_metadata(config=None):
    config = codec_config(config)
    spec = dict(
        representation=REPRESENTATION,
        grid_policy=GRID_POLICY,
        fields=list(FIELDS),
        config=config,
        specials=dict(pad=PAD, unknown=UNK, mask=MASK, start=START),
        value_offset=OFFSET,
        time_codebook=dict(
            linear_max_quarters=4,
            resolution=config["resolution"],
            log_bins_per_octave=48,
            log_steps=384,
            overflow_quarters=1024,
            rounding="nearest_ties_up_rational_linear",
        ),
        tempo=dict(min_qpm=8, max_qpm=512, bins_per_octave=24),
        meter=dict(max_numerator=255, max_denominator_power=6),
        max_position_quarters=1024,
    )
    spec["sha256"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return spec


def _fraction(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("Musical times must be finite numbers")
    if not math.isfinite(value):
        raise ValueError("Musical times must be finite numbers")
    return value if isinstance(value, Fraction) else Fraction(str(value))


def event_quarter(event, source):
    """Prefer original integer ticks; recovered metadata may have fractional 960 ticks."""
    if "original_tick" in event and source.get("original_ticks_per_quarter"):
        return _fraction(event["original_tick"]) / source["original_ticks_per_quarter"]
    if "tick" in event:
        return _fraction(event["tick"]) / source["ticks_per_quarter"]
    if "tick_960" in event:
        return _fraction(event["tick_960"]) / 960
    raise ValueError("Timing event needs tick or calibrated tick_960")


class MusicalClock:
    """Exact bar positions and piecewise tempo integration, independent of note order."""

    def __init__(self, source):
        if not isinstance(source, dict):
            raise ValueError("Octuple inputs require source timing metadata")
        ppq = source.get("ticks_per_quarter")
        if type(ppq) is not int or ppq < 1:
            raise ValueError("Positive integer ticks_per_quarter required")
        if source.get("seconds_status") == "unresolved_tempo_metadata":
            raise ValueError("Unresolved tempo metadata")
        self.ppq = ppq
        self.signatures, self.origins, self.bases = [], [], []
        self.diagnostics = dict(redundant_signatures=0, off_bar_changes=0)
        signatures = {}
        for event in source.get("time_signatures", []):
            q = event_quarter(event, source)
            n, d = event["numerator"], event["denominator"]
            if (
                q < 0
                or type(n) is not int
                or not 1 <= n <= 255
                or type(d) is not int
                or d not in (1, 2, 4, 8, 16, 32, 64)
            ):
                raise ValueError("Unsupported explicit MIDI meter")
            if q in signatures and signatures[q] != (n, d):
                raise ValueError("Conflicting explicit MIDI meter")
            signatures[q] = n, d
        for q, signature in sorted(signatures.items()):
            if self.signatures and signature == self.signatures[-1]:
                self.diagnostics["redundant_signatures"] += 1
                continue
            base = 0
            if self.origins:
                n, d = self.signatures[-1]
                elapsed = (q - self.origins[-1]) / Fraction(4 * n, d)
                self.diagnostics["off_bar_changes"] += int(elapsed.denominator != 1)
                base = self.bases[-1] + math.ceil(elapsed)
            self.origins.append(q)
            self.signatures.append(signature)
            self.bases.append(base)
        # An explicit first-bar origin is optional and never inferred from lyrics.
        if source.get("bar_origin_tick") is not None:
            if not self.origins:
                raise ValueError("Bar origin requires an explicit signature")
            origin = _fraction(source["bar_origin_tick"]) / ppq
            if origin > self.origins[0]:
                raise ValueError("First bar origin must not follow the first signature")
            self.origins[0] = origin
            for j in range(1, len(self.origins)):
                n, d = self.signatures[j - 1]
                self.bases[j] = self.bases[j - 1] + math.ceil(
                    (self.origins[j] - self.origins[j - 1]) / Fraction(4 * n, d)
                )
        tempos = {}
        for event in source.get("tempo_map", []):
            q, tempo = event_quarter(event, source), _fraction(event["tempo"])
            if q < 0 or tempo <= 0:
                raise ValueError("Invalid MIDI tempo")
            if q in tempos and tempos[q] != tempo:
                raise ValueError("Conflicting MIDI tempo events")
            tempos[q] = tempo
        self.explicit_tempo_start = min(tempos) if tempos else None
        tempos.setdefault(Fraction(0), Fraction(500000))
        self.tempo_times = sorted(tempos)
        self.tempos = [tempos[q] for q in self.tempo_times]
        self.seconds = [Fraction(0)]
        for j in range(1, len(self.tempo_times)):
            self.seconds.append(
                self.seconds[-1]
                + (self.tempo_times[j] - self.tempo_times[j - 1]) * self.tempos[j - 1] / 1000000
            )

    def meter(self, quarter):
        j = bisect_right(self.origins, quarter) - 1
        if j < 0:
            raise ValueError("Missing explicit meter at a melody note")
        n, d = self.signatures[j]
        length = Fraction(4 * n, d)
        elapsed = quarter - self.origins[j]
        bars = elapsed // length
        return self.bases[j] + bars, elapsed - bars * length, n, d

    def tempo(self, quarter):
        j = bisect_right(self.tempo_times, quarter) - 1
        return float(60000000 / self.tempos[j])

    def seconds_at(self, quarter):
        j = bisect_right(self.tempo_times, quarter) - 1
        if j < 0:
            raise ValueError("Negative MIDI time")
        return float(self.seconds[j] + (quarter - self.tempo_times[j]) * self.tempos[j] / 1000000)


def canonical_notes(lines, source):
    clock = MusicalClock(source)
    notes, line_ids = [], []
    previous_end = None
    for line_id, line in enumerate(lines, 1):
        m = line["melody"]
        pitches, ticks, durations = (
            m[k] for k in ("midi_pitches", "onset_ticks", "note_duration_ticks")
        )
        if not pitches or not len(pitches) == len(ticks) == len(durations):
            raise ValueError("Canonical melody arrays must be nonempty and aligned")
        for pitch, tick, duration in zip(pitches, ticks, durations, strict=True):
            if type(pitch) is not int or not 0 <= pitch <= 127:
                raise ValueError("MIDI pitch must be an integer in 0..127")
            onset, dur = _fraction(tick) / clock.ppq, _fraction(duration) / clock.ppq
            if onset < 0 or dur <= 0 or (previous_end is not None and onset < previous_end):
                raise ValueError("Invalid or overlapping monophonic melody notes")
            notes.append((pitch, onset, dur))
            line_ids.append(line_id)
            previous_end = onset + dur
    if not notes:
        raise ValueError("Empty canonical melody")
    return notes, line_ids, clock


def canonical_meter(lines, source):
    notes, _, clock = canonical_notes(lines, source)
    result = []
    for _, onset, _ in notes:
        _, position, n, d = clock.meter(onset)
        result.append(dict(numerator=n, denominator=d, beat_position=float(position * d / 4)))
    return result, clock.diagnostics


def _time_code(value, centers, resolution, *, positive=False):
    if value <= 4:
        j = max(1 if positive else 0, math.floor(value * resolution + Fraction(1, 2)))
        return OFFSET + j, float((value - Fraction(j, resolution)) * resolution)
    value = float(value)
    if value > centers[-1]:
        return OFFSET + len(centers), (value - 1024) / 1024
    lo = 1 if positive else 0
    j = max(lo, min(len(centers) - 1, int(np.searchsorted(centers, value))))
    if j > lo and value - centers[j - 1] < centers[j] - value:
        j -= 1
    center = centers[j]
    width = 1 / resolution if center <= 4 else center * (2 ** (1 / 48) - 1)
    return OFFSET + j, (value - center) / width


def encode_octuple(lines, source, config=None):
    config = codec_config(config)
    q = config["resolution"]
    centers = time_centers(q)
    notes, line_ids, clock = canonical_notes(lines, source)
    tuples, residuals, meters = [], [], []
    first_bar = clock.meter(notes[0][1])[0]
    for i, (pitch, onset, duration) in enumerate(notes):
        bar, position, n, d = clock.meter(onset)
        pos = min(math.floor(q * position + Fraction(1, 2)), q * 4 * n // d - 1)
        dur_code, dur_residual = _time_code(duration, centers, q, positive=True)
        gap_code, gap_residual, interval = START, 0.0, START
        if i:
            prev_pitch, prev_onset, prev_duration = notes[i - 1]
            gap_code, gap_residual = _time_code(onset - prev_onset - prev_duration, centers, q)
            interval = OFFSET + pitch - prev_pitch + 127
        tempo = clock.tempo(onset)
        tempo_code = (
            145
            if tempo < 8
            else 146
            if tempo > 512
            else math.floor(24 * math.log2(tempo / 8) + 0.5)
        )
        tuples.append(
            [
                OFFSET + bar - first_bar,
                OFFSET + pos,
                OFFSET + pitch,
                interval,
                dur_code,
                gap_code,
                OFFSET + (n - 1) * 7 + int(math.log2(d)),
                OFFSET + tempo_code,
            ]
        )
        residuals.append([float(q * position - pos), dur_residual, gap_residual])
        meters.append(dict(numerator=n, denominator=d, beat_position=float(position * d / 4)))
    return dict(
        melody_features=np.asarray(tuples, dtype=np.int64),
        timing_residuals=np.asarray(residuals, dtype=np.float32),
        note_line_ids=line_ids,
        line_count=len(lines),
        meter=meters,
    )
