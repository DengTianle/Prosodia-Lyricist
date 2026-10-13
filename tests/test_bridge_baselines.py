"""Duration thresholds, reference-count alignment, and paired model comparisons."""

import json

import numpy as np
import pytest

from prosodia_lyricist.bridge_baselines import evaluate_duration_baselines
from prosodia_lyricist.bridge_data import (
    BridgeDataset,
    bridge_collate,
    encode_bridge_source,
    encode_bridge_targets,
)
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import SCHEMA_VERSION


def lines_for(durations, lengths):
    return [{
        "melody": {
            "midi_pitches": [60] * len(notes),
            "onset_seconds": [index * 20 + i for i in range(len(notes))],
            "note_duration_seconds": notes,
        },
        "syllables": [{"stress": "weak", "length": length} for length in labels],
    } for index, (notes, labels) in enumerate(zip(durations, lengths, strict=True))]


def example(durations, lengths, song_mean=None):
    lines = lines_for(durations, lengths)
    durations = np.concatenate(durations)
    return {
        **encode_bridge_source(lines), "labels": encode_bridge_targets(lines, 8),
        "note_duration_seconds": durations,
        "song_mean_note_duration_seconds": (
            float(durations.mean()) if song_mean is None else song_mean
        ),
    }


@pytest.mark.parametrize("lengths,correct", [
    ([["short", "long"], ["short"], ["short", "long"]],
     {"global": 3, "song": 4, "window": 4, "line": 6}),
    ([["short", "short"], ["long"], ["short", "long"]],
     {"global": 5, "song": 4, "window": 6, "line": 4}),
])
def test_means_are_note_weighted_and_respect_lines_windows_tail_and_ties(lengths, correct):
    examples = [example([[1, 3], [5], [7, 9]], lengths), example([[1]], [["short"]])]
    scores = evaluate_duration_baselines(examples, lines_per_window=2)
    assert scores["global_mean_note_duration_seconds"] == pytest.approx(26 / 6)
    for subset in ("all_ipa_slots", "equal_note_ipa_count"):
        assert scores[subset]["compared_slots"] == 6
        for scope, count in correct.items():
            result = scores[subset]["baselines"][scope]
            assert result["correct_slots"] == count
            assert result["length_accuracy"] == pytest.approx(count / 6)


def test_missing_notes_are_wrong_extra_notes_trimmed_and_model_comparison_uses_same_subset():
    examples = [example([[1, 3], [1, 3, 5], [2]],
                        [["short", "long", "long"], ["short"], ["short"]])]
    generation = {
        "phrases": 3, "compared_slots": 5, "length_accuracy": 0.6,
        "equal_note_ipa_count": {"phrases": 1, "compared_slots": 1, "length_accuracy": 0.0},
    }
    scores = evaluate_duration_baselines(examples, lines_per_window=2, generation=generation)
    assert scores["missing_note_slots"] == 1
    assert scores["extra_notes_ignored"] == 2
    full, equal = scores["all_ipa_slots"], scores["equal_note_ipa_count"]
    assert full["baselines"]["line"]["length_accuracy"] == 4 / 5
    assert full["baselines"]["line"]["minus_model_length_accuracy"] == pytest.approx(0.2)
    assert equal["baselines"]["line"]["length_accuracy"] == 1.0
    assert equal["baselines"]["line"]["minus_model_length_accuracy"] == 1.0
    assert full["baselines"]["always_long"]["length_accuracy"] == 2 / 5
    assert full["baselines"]["always_short"]["length_accuracy"] == 3 / 5
    generation["compared_slots"] += 1
    with pytest.raises(ValueError, match="coverage differs"):
        evaluate_duration_baselines(examples, lines_per_window=2, generation=generation)


def test_empty_equal_count_subset_is_null_and_serializable():
    scores = evaluate_duration_baselines(
        [example([[1]], [["short", "long"]])], lines_per_window=2
    )
    assert scores["equal_note_ipa_count"]["compared_slots"] == 0
    assert scores["equal_note_ipa_count"]["baselines"]["line"]["length_accuracy"] is None
    json.dumps(scores, allow_nan=False)


def test_legacy_window_limit_keeps_whole_song_mean_but_global_mean_uses_retained_notes(tmp_path):
    song = {"song_id": "test-song", "lines": lines_for(
        [[1, 3], [5], [7, 9]], [["short", "long"], ["short"], ["short", "long"]]
    )}
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(song) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "config": {"include_melody": True, "stress_source": "ipa", "max_syllables": 8},
        "songs": {"test-song": {"split": "test", "group": "test-song"}},
        "sha256": {"test.jsonl": file_sha256(path)},
    }))
    dataset = BridgeDataset(tmp_path, "test", lines_per_window=2, max_notes=8, limit=1)
    scores = evaluate_duration_baselines(dataset.examples, lines_per_window=2)
    assert scores["global_mean_note_duration_seconds"] == 3
    assert dataset[0]["song_mean_note_duration_seconds"] == 5
    assert scores["all_ipa_slots"]["compared_slots"] == 3
    batch = bridge_collate(dataset.examples)
    assert "note_duration_seconds" not in batch
    assert "song_mean_note_duration_seconds" not in batch
