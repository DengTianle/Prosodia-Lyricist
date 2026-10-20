"""Build soft melody conditioning and an explicitly diagnostic hard-template preview."""

import torch

from .bridge_data import SLOT, decode_bridge_tokens
from .end_to_end_data import EndToEndCollator, encode_song
from .end_to_end_model import condition_inputs
from .midi import midi_melody_record


@torch.inference_mode()
def condition_midi(
    model, tokenizer, midi, *, title, track, device, max_source_length, template_mode="soft"
):
    record = midi_melody_record(midi, title=title, track=track, **(
        {"phrase_policy": "repo"} if model.prosody_mode == "actual" else {}
    ))
    bridge = model.bridge
    if len(record["lines"]) > bridge.max_song_lines:
        raise ValueError("MIDI exceeds the checkpoint's phrase limit")
    example = encode_song(
        record,
        tokenizer,
        max_syllables=bridge.max_syllables,
        encoder_lines_per_window=bridge.encoder_lines_per_window,
        count_source="notes",
        with_targets=False,
        decoder_mode=model.decoder_mode,
        strength_labels=bridge.strength_labels,
    )
    if len(example["source"]["input_ids"]) > max_source_length:
        raise ValueError("MIDI exceeds the checkpoint's source token limit")
    batch = {
        k: v.to(device) for k, v in EndToEndCollator(tokenizer.pad_token_id)([example]).items()
    }
    source = model.condition_source(**condition_inputs(batch), template_mode=template_mode)
    predictions = decode_bridge_tokens(source.sequences[0].tolist(), pairs=bridge.pairs)
    records = [
        {**line, **predicted, "line_id": index, "stress_source": "learned"}
        for index, (line, predicted) in enumerate(zip(record["lines"], predictions, strict=True))
    ]
    slots = source.bridge.skeleton["tokens"][0].eq(SLOT)
    metadata = {
        "conditioning": "parallel_soft_template",
        "mode": template_mode,
        "temperature": model.temperature,
        "prosody_mode": model.prosody_mode,
        "stress_labels": list(bridge.strength_labels),
        "unmarked_notes_omitted": record.get("unmarked_notes_omitted", 0),
        "count_sources": ["notes"] * len(records),
        "note_counts": batch["syllable_counts"][0].tolist(),
        "syllable_counts": batch["syllable_counts"][0].tolist(),
        "stress_probabilities": source.stress_probabilities[0, slots].tolist(),
        "length_probabilities": source.length_probabilities[0, slots].tolist(),
        "template_role": "argmax_preview" if template_mode == "soft" else "model_input",
        "melody_provenance": bridge.provenance,
    }
    return records, {k: v for k, v in example["source"].items() if k != "slot_indices"}, (
        source.inputs_embeds
    ), metadata
