"""Guard the audit's label false positives and statistical denominators."""

import json

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


@pytest.mark.parametrize("note_limit,retained", [(5, 0), (6, 1)])
def test_audit_reports_complete_songs_and_global_limits(
    tmp_path, monkeypatch, note_limit, retained
):
    from prosodia_lyricist.audit_bridge import main
    from prosodia_lyricist.melody_checkpoint import file_sha256

    manifest = dict(
        config=dict(stress_source="ipa", include_melody=True, max_syllables=8),
        songs={},
        sha256={},
        counts={},
    )
    for split in ("train", "valid", "test"):
        lines = [
            dict(
                line_id=i,
                text="la",
                start=i * 3,
                end=(i + 1) * 3,
                melody=dict(
                    midi_pitches=[60, 62, 64],
                    onset_seconds=[i * 3 + j for j in range(3)],
                    note_duration_seconds=[1, 1, 1],
                ),
                syllables=[dict(stress="strong", length="long")],
                sung_syllables=[dict(note_count=3)],
                words=[dict(syllable_count=1)],
            )
            for i in range(2)
        ]
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(dict(song_id=split, title=split, lines=lines)) + "\n")
        manifest["songs"][split] = dict(split=split)
        manifest["sha256"][path.name] = file_sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "rejected.jsonl").write_text("")
    output = tmp_path / "audit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--prepared-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--max-notes",
            str(note_limit),
        ],
    )
    main()
    report = json.loads((output / "summary.json").read_text())
    assert report["groups"]["train"]["retained_songs"]["rows"] == retained
    assert report["groups"]["train"]["skipped_songs"] == 1 - retained
    assert (output / "songs.csv").exists()
    assert not (output / "windows.csv").exists()
