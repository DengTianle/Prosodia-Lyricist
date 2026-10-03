"""Convert MIDI notes to learned templates for an unchanged template decoder."""

from pathlib import Path

import torch

from .bridge_data import (
    bridge_collate,
    decode_bridge_tokens,
    encode_bridge_song,
    encode_bridge_source,
)
from .bridge_model import ProsodyBridge
from .features import MAX_LINES
from .midi import midi_melody_record


@torch.inference_mode()
def predict_templates(
    checkpoint,
    midi,
    *,
    title="",
    track=0,
    device="cpu",
    max_syllables=None,
):
    model = ProsodyBridge.load(checkpoint).to(device).eval()
    if max_syllables is not None and model.max_syllables > max_syllables:
        raise ValueError("Bridge syllable limit exceeds the template decoder's limit")
    source = midi_melody_record(midi, title=title, track=track)
    if len(source["lines"]) > MAX_LINES:
        raise ValueError(f"Template decoder supports at most {MAX_LINES} MIDI phrases")
    for index, line in enumerate(source["lines"]):
        count = len(line["melody"]["midi_pitches"])
        if count > model.max_syllables:
            raise ValueError(
                f"MIDI phrase {index} has {count} notes; bridge slot limit is "
                f"{model.max_syllables}; no truncation"
            )
    records = []
    windows = 0
    width = len(source["lines"]) if model.bridge_scope == "song" else model.lines_per_window
    for offset in range(0, len(source["lines"]), width):
        lines = source["lines"][offset : offset + width]
        encoded = (
            encode_bridge_song(lines, model.encoder_lines_per_window)
            if model.bridge_scope == "song" else encode_bridge_source(lines)
        )
        if model.bridge_scope == "window" and len(encoded["note_line_ids"]) > model.max_notes:
            raise ValueError("MIDI window exceeds bridge note limit; no truncation")
        inputs = {key: value.to(device) for key, value in bridge_collate([encoded]).items()}
        result = model.generate(**inputs)
        predicted = decode_bridge_tokens(result.sequences[0].tolist())
        for index, (line, template) in enumerate(zip(lines, predicted, strict=True), offset):
            records.append({**line, **template, "line_id": index, "stress_source": "learned"})
        windows += (
            len(lines) + model.encoder_lines_per_window - 1
        ) // model.encoder_lines_per_window
    return records, {
        "checkpoint": str(Path(checkpoint).resolve()),
        "melody_provenance": model.provenance,
        "lines_per_window": model.lines_per_window,
        "encoder_lines_per_window": model.encoder_lines_per_window,
        "bridge_scope": model.bridge_scope,
        "encoder_windows": windows,
        "max_syllables": model.max_syllables,
        "max_notes": model.max_notes,
        "max_song_notes": model.max_song_notes if model.bridge_scope == "song" else None,
        "max_song_lines": model.max_song_lines if model.bridge_scope == "song" else None,
        "max_target_length": model.max_target_length,
        "target_scheme": model.target_scheme,
        "output_design": model.output_design,
        "count_sources": ["notes"] * len(records),
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
