"""MIDI note conditioning with one shared four-stream decoder context."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .melody_data import MelodyCollator, encode_melody_source
from .melody_model import MelodyBart
from .runtime import seed_everything, select_device


def midi_melody_record(path, *, track=0, title=""):
    import miditoolkit

    midi = miditoolkit.MidiFile(str(path))
    if not 0 <= track < len(midi.instruments):
        raise ValueError("MIDI melody track is out of range")
    instrument = midi.instruments[track]
    notes = sorted(instrument.notes, key=lambda n: (n.start, n.end))
    if instrument.is_drum or not notes:
        raise ValueError("Select a non-empty, non-drum melody track")
    if any(n.start < 0 or n.end <= n.start for n in notes):
        raise ValueError("MIDI has invalid note timing")
    if any(a.end > b.start for a, b in zip(notes, notes[1:])):
        raise ValueError("MIDI melody track must be monophonic")
    boundaries = sorted({m.time for m in midi.markers})
    if not boundaries:
        raise ValueError("MIDI needs phrase-end markers matching training lyric lines")
    if boundaries[-1] < notes[-1].end:
        boundaries.append(notes[-1].end)
    times = midi.get_tick_to_time_mapping()
    lines, cursor = [], 0
    for boundary in boundaries:
        group = []
        # Preserve this project's phrase-end convention, including exact-boundary onsets.
        while cursor < len(notes) and notes[cursor].start <= boundary:
            group.append(notes[cursor])
            cursor += 1
        if group:
            lines.append(
                {
                    "melody": {
                        "midi_pitches": [n.pitch for n in group],
                        "onset_seconds": [float(times[n.start]) for n in group],
                        "note_duration_seconds": [
                            float(times[n.end] - times[n.start]) for n in group
                        ],
                    }
                }
            )
    return {"title": title, "lines": lines}


@torch.inference_mode()
def infer_melody(
    checkpoint,
    midi,
    *,
    title="",
    track=0,
    device="auto",
    temperature=1.0,
    top_k=3,
    max_new_tokens=None,
    seed=1234,
    return_explanations=False,
    prosody_correction=True,
):
    seed_everything(seed)
    checkpoint = Path(checkpoint)
    run = json.loads((checkpoint / "run.json").read_text())
    model = MelodyBart.load(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    limits = run["config"]["model"]
    target_limit = min(limits["max_target_length"], model.bart.config.max_position_embeddings)
    max_new_tokens = target_limit - 1 if max_new_tokens is None else max_new_tokens
    if not 1 <= max_new_tokens < target_limit:
        raise ValueError("Generation budget exceeds checkpoint target limit")
    record = midi_melody_record(midi, track=track, title=title)
    source = encode_melody_source(record, tokenizer, **model.source_config)
    if len(source["input_ids"]) > limits["max_source_length"]:
        raise ValueError("Complete MIDI source exceeds checkpoint limit; supply a paragraph MIDI")
    device = select_device(device)
    model.to(device).eval()
    inputs = {
        key: value.to(device)
        for key, value in MelodyCollator(tokenizer.pad_token_id)([source]).items()
    }
    result = model.generate(
        **inputs,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=top_k > 1,
        temperature=temperature,
        top_k=top_k,
        prosody_correction=prosody_correction,
    )
    explanation = result.explanations[0]
    if not return_explanations:
        return explanation["lines"]
    explanation.update(
        {
            "conditioning": "melody",
            "source": record,
            "source_config": model.source_config,
            "source_tokens": len(source["input_ids"]),
            "input_phrases": len(record["lines"]),
            "encoder_windows": len(source["windows"]),
            "streams": {
                name: getattr(result, name)[0].tolist()
                for name in (
                    "sequences",
                    "syllable_ids",
                    "stress_ids",
                    "length_ids",
                    "attention_mask",
                )
            },
        }
    )
    return explanation
