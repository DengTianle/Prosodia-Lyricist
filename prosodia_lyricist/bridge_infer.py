"""Convert MIDI notes to learned templates for an unchanged template decoder."""

from pathlib import Path

import torch

from .bridge_data import bridge_collate, decode_bridge_tokens, encode_bridge_source
from .bridge_model import ProsodyBridge
from .features import MAX_LINES
from .midi import midi_melody_record


@torch.inference_mode()
def predict_templates(checkpoint, midi, *, title="", track=0, device="cpu", max_syllables=None):
    model = ProsodyBridge.load(checkpoint).to(device).eval()
    if max_syllables is not None and model.max_syllables > max_syllables:
        raise ValueError("Bridge syllable limit exceeds the template decoder's limit")
    source = midi_melody_record(midi, title=title, track=track)
    if len(source["lines"]) > MAX_LINES:
        raise ValueError(f"Template decoder supports at most {MAX_LINES} MIDI phrases")
    records, forced = [], []
    windows = 0
    for offset in range(0, len(source["lines"]), model.lines_per_window):
        lines = source["lines"][offset : offset + model.lines_per_window]
        encoded = encode_bridge_source(lines)
        if len(encoded["note_line_ids"]) > model.max_notes:
            raise ValueError("MIDI window exceeds bridge note limit; no truncation")
        inputs = {key: value.to(device) for key, value in bridge_collate([encoded]).items()}
        result = model.generate(**inputs)
        predicted = decode_bridge_tokens(result.sequences[0].tolist())
        for line, template in zip(lines, predicted, strict=True):
            records.append({**line, **template, "stress_source": "learned"})
        forced.extend(offset + index for index in result.forced_line_endings[0])
        windows += 1
    return records, {
        "checkpoint": str(Path(checkpoint).resolve()),
        "melody_provenance": model.provenance,
        "lines_per_window": model.lines_per_window,
        "encoder_windows": windows,
        "max_syllables": model.max_syllables,
        "max_notes": model.max_notes,
        "forced_line_endings": forced,
        "note_counts": [len(line["melody"]["midi_pitches"]) for line in records],
        "syllable_counts": [len(line["syllables"]) for line in records],
        "decoding": "greedy",
    }
