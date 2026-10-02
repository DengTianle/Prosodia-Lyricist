"""Guard the audit's label false positives and statistical denominators."""

import pytest

from prosodia_lyricist.audit_bridge import marker_flags, summarize


@pytest.mark.parametrize("text", ["[chorus]", "[Chorus: 2]", "(repeat chorus x2)"])
def test_standalone_chorus_label(text):
    flags = marker_flags(text)
    assert flags["bracketed_label"] and flags["standalone_label"]
    assert flags["kinds"] == ["chorus_or_refrain"]


def test_label_variants_and_inline_text():
    assert marker_flags("[guitar-solo]")["kinds"] == ["instrumental_or_solo"]
    assert marker_flags("[ instrumental ]")["standalone_label"]
    assert marker_flags("[chorus] sing along")["inline_label"]
    assert not marker_flags("[chorus] sing along")["standalone_label"]
    assert marker_flags("chorus")["bare_label_candidate"]
    assert marker_flags("*bass-solo*")["decorated_label"]
    assert not marker_flags("*bass-solo*")["bare_label_candidate"]
    assert marker_flags("sing along *solo* sing again")["inline_label"]


@pytest.mark.parametrize(
    "text",
    [
        "when they bring that chorus in",
        "[say hello]",
        "[i'm walking on pins and needles]",
        "vocally and instrumentally",
        "[the chorus yelled unity]",
    ],
)
def test_lyrics_and_brackets_are_not_automatically_section_labels(text):
    assert not marker_flags(text)["kinds"]


def test_mismatch_song_denominators_and_sung_distinction():
    rows = [
        {"song_id": "a", "notes": 5, "ipa": 3, "sung": 3, "duration_seconds": 1},
        {"song_id": "a", "notes": 3, "ipa": 4, "sung": 3, "duration_seconds": 1},
        {"song_id": "b", "notes": 1, "ipa": 1, "sung": 1, "duration_seconds": 1},
    ]
    summary = summarize(rows)
    comparison = summary["mismatches"]["ipa_minus_notes"]
    assert comparison["different"]["rows"] == 2
    assert comparison["different"]["songs"] == 1
    assert comparison["different"]["pct_songs"] == 50
    assert comparison["mean_signed_difference"] == pytest.approx(-1 / 3)
    assert comparison["mean_absolute_difference"] == 1
    assert summary["mismatches"]["sung_minus_notes"]["different"]["rows"] == 1
    assert summary["mismatches"]["ipa_minus_sung"]["different"]["rows"] == 1
