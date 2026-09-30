"""Convert MIDI notes to learned templates for an unchanged template decoder."""

import json
from pathlib import Path

import torch

from .bridge_data import bridge_collate, decode_bridge_tokens, encode_bridge_source
from .bridge_model import ProsodyBridge
from .features import MAX_LINES
from .midi import midi_melody_record


def skeleton_counts(skeleton, line_count, max_syllables):
    """Public line IDs are ordered, zero-based MIDI phrase indices; missing counts are learned."""
    if skeleton is None:
        return [0] * line_count
    if isinstance(skeleton, (str, Path)):
        skeleton = json.loads(Path(skeleton).read_text(encoding="utf-8"))
    if not isinstance(skeleton, dict) or set(skeleton) != {"lines"}:
        raise ValueError("Bridge skeleton must be a JSON object containing a lines list")
    lines = skeleton["lines"]
    if not isinstance(lines, list) or len(lines) != line_count:
        raise ValueError("Bridge skeleton must contain one entry per MIDI phrase")
    counts = []
    for index, line in enumerate(lines):
        if not isinstance(line, dict) or set(line) - {"line_id", "syllable_count"}:
            raise ValueError("Skeleton entries accept line_id and optional syllable_count")
        if type(line.get("line_id")) is not int or line["line_id"] != index:
            raise ValueError("Skeleton line_id must follow MIDI phrase order: 0, 1, 2, ...")
        count = line.get("syllable_count")
        if count is not None and (type(count) is not int or not 1 <= count <= max_syllables):
            raise ValueError(
                f"Skeleton syllable_count must be null or an integer in 1..{max_syllables}"
            )
        counts.append(0 if count is None else count)
    return counts


@torch.inference_mode()
def predict_templates(
    checkpoint,
    midi,
    *,
    title="",
    track=0,
    device="cpu",
    max_syllables=None,
    skeleton=None,
):
    model = ProsodyBridge.load(checkpoint).to(device).eval()
    if max_syllables is not None and model.max_syllables > max_syllables:
        raise ValueError("Bridge syllable limit exceeds the template decoder's limit")
    source = midi_melody_record(midi, title=title, track=track)
    if len(source["lines"]) > MAX_LINES:
        raise ValueError(f"Template decoder supports at most {MAX_LINES} MIDI phrases")
    requested = skeleton_counts(skeleton, len(source["lines"]), model.max_syllables)
    records, predicted_counts = [], []
    windows = 0
    for offset in range(0, len(source["lines"]), model.lines_per_window):
        lines = source["lines"][offset : offset + model.lines_per_window]
        encoded = encode_bridge_source(lines)
        if len(encoded["note_line_ids"]) > model.max_notes:
            raise ValueError("MIDI window exceeds bridge note limit; no truncation")
        inputs = {key: value.to(device) for key, value in bridge_collate([encoded]).items()}
        result = model.generate(
            **inputs,
            syllable_counts=torch.tensor([requested[offset : offset + len(lines)]], device=device),
        )
        predicted = decode_bridge_tokens(result.sequences[0].tolist())
        for index, (line, template) in enumerate(zip(lines, predicted, strict=True), offset):
            records.append({**line, **template, "line_id": index, "stress_source": "learned"})
        predicted_counts.extend(result.predicted_syllable_counts[0].tolist())
        windows += 1
    return records, {
        "checkpoint": str(Path(checkpoint).resolve()),
        "melody_provenance": model.provenance,
        "lines_per_window": model.lines_per_window,
        "encoder_windows": windows,
        "max_syllables": model.max_syllables,
        "max_notes": model.max_notes,
        "target_scheme": "line_skeleton_v2",
        "requested_syllable_counts": [count or None for count in requested],
        "predicted_syllable_counts": predicted_counts,
        "count_sources": ["provided" if count else "predicted" for count in requested],
        "skeleton": {
            "lines": [
                {"line_id": line["line_id"], "syllable_count": len(line["syllables"])}
                for line in records
            ]
        },
        "note_counts": [len(line["melody"]["midi_pitches"]) for line in records],
        "syllable_counts": [len(line["syllables"]) for line in records],
        "decoding": "greedy_skeleton",
    }
