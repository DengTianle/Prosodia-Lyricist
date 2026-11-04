"""Convert MIDI notes to learned templates for an unchanged template decoder."""

from pathlib import Path

import torch

from .bridge_data import (
    bridge_collate,
    decode_bridge_tokens,
    encode_bridge_song,
)
from .bridge_model import ProsodyBridge
from .ipa import REPO_RULES
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
    return predict_midi_templates(
        model,
        midi,
        title=title,
        track=track,
        device=device,
        max_syllables=max_syllables,
        checkpoint=checkpoint,
    )


@torch.inference_mode()
def predict_midi_templates(
    model,
    midi,
    *,
    title="",
    track=0,
    device="cpu",
    max_syllables=None,
    checkpoint=None,
    phrase_policy=None,
):
    """Predict with an already loaded, eval-mode bridge (reusable across songs)."""
    if max_syllables is not None and model.max_syllables > max_syllables:
        raise ValueError("Bridge syllable limit exceeds the template decoder's limit")
    mode = "actual" if model.prosody_rules == REPO_RULES else "paper"
    source = midi_melody_record(
        midi,
        title=title,
        track=track,
        phrase_policy=phrase_policy or "complete",
        canonical=model.canonical_timing,
    )
    for index, line in enumerate(source["lines"]):
        count = len(line["melody"]["midi_pitches"])
        if not model.predict_note_roles and count > model.max_syllables:
            raise ValueError(
                f"MIDI phrase {index} has {count} notes; bridge slot limit is "
                f"{model.max_syllables}; no truncation"
            )
    encoded = encode_bridge_song(source["lines"], representation=model.representation,
                                 source=source.get("source"), octuple_config=model.octuple_config,
                                 canonical_timing=model.canonical_timing)
    inputs = {key: value.to(device) for key, value in bridge_collate([encoded]).items()}
    result = model.generate(**inputs)
    predicted = decode_bridge_tokens(result.sequences[0].tolist(), pairs=model.pairs)
    records = [
        {**line, **template, "line_id": index, "stress_source": "learned"}
        for index, (line, template) in enumerate(zip(source["lines"], predicted, strict=True))
    ]
    if records and mode == "actual":
        records[-1]["unmarked_notes_omitted"] = source.get("unmarked_notes_omitted", 0)
    return records, {
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else None,
        "melody_provenance": model.provenance,
        "prosody_mode": mode,
        "stress_labels": list(model.strength_labels),
        "unmarked_notes_omitted": source.get("unmarked_notes_omitted", 0),
        "bridge_scope": "song",
        "source_scheme": model.source_scheme,
        "meter_conditioning": model.meter_conditioning,
        "observed_meter_notes": int(inputs["meter_mask"].sum()),
        "contrastive_fusion": model.contrastive_config is not None,
        "max_syllables": model.max_syllables,
        "max_song_notes": model.max_song_notes,
        "max_song_lines": model.max_song_lines,
        "max_target_length": model.max_target_length,
        "target_scheme": model.target_scheme,
        "output_design": model.output_design,
        "query_mode": model.query_mode,
        "query_attention": model.query_attention,
        "prosody_rules": model.prosody_rules,
        "count_sources": ["predicted_syllable_starts" if model.predict_note_roles else "notes"]
        * len(records),
        "phrase_source": "supplied_markers",
        "role_threshold": model.role_threshold if model.predict_note_roles else None,
        "note_start_mask": result.note_start_mask[0].tolist(),
        "skeleton": {
            "lines": [
                {"line_id": line["line_id"], "syllable_count": len(line["syllables"])}
                for line in records
            ]
        },
        "note_counts": [len(line["melody"]["midi_pitches"]) for line in records],
        "syllable_counts": [len(line["syllables"]) for line in records],
        "decoding": "note_roles_and_heads" if model.predict_note_roles else "note_heads",
    }
