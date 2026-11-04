import copy
import json
import random

import pytest

from prosodia_lyricist import pipeline_eval, pipeline_reports


def labels(count=4):
    return [
        {"word": "word", "ipa": "æ", "stress": "strong", "length": "short"} for _ in range(count)
    ]


@pytest.fixture
def saved_report():
    reference = [
        {"text": "word word word word", "syllables": labels()},
        {"text": "reference only", "syllables": labels(2)},
    ]
    midi = {
        "syllables": [
            {
                "stress": "strong",
                "length": "short",
                "note": {
                    "pitch": 60,
                    "start_tick": i * 240,
                    "end_tick": (i + 1) * 240,
                    "measure": 1,
                    "quarter_beat": i / 2 + 1,
                    "length_threshold_ticks": 240,
                },
            }
            for i in range(4)
        ],
        "length_rule": "onset_bucket_8_quarters",
        "length_threshold_ticks": 240,
    }
    bridge = {"syllables": labels(), "melody": {"midi_pitches": [60] * 4}}
    generated = {"text": "word word word word", "syllables": labels(), "error": None}

    def method(template):
        return {
            "template": [template],
            "lyrics": [generated["text"]],
            "generated_prosody": [generated],
            "pronunciation_failures": 0,
            "generation": {
                "seed": 3,
                "hit_token_limit": False,
                "no_repeat_ngram_size": 0,
                "repetition_penalty": 1,
            },
            "template_reference": pipeline_eval.compare_prosody([template], reference),
            "lyrics_template": pipeline_eval.compare_prosody([generated], [template]),
            "lyrics_reference": pipeline_eval.compare_prosody([generated], reference),
            "generated_perplexity": {"perplexity": 2.0, "token_count": 5},
            "reference_perplexity": {"perplexity": 3.0, "token_count": 8},
        }

    return {
        "schema_version": 1,
        "prosody_mode": "actual",
        "generation": {"seed": 1234},
        "checkpoint": "/missing/checkpoint",
        "methods": {
            "actual": {"predictor": {"kind": "midi", "stress_source": "actual"}},
            "bridge": {"predictor": {"kind": "bridge", "checkpoint": "/missing/bridge"}},
        },
        "per_song": [
            {
                "song_id": f"song_{i}",
                "title": f"Song {i}",
                "midi": "/missing/song.mid",
                "reference": copy.deepcopy(reference),
                "methods": {
                    "actual": method(copy.deepcopy(midi)),
                    "bridge": method(copy.deepcopy(bridge)),
                },
            }
            for i in range(5)
        ],
    }


def test_random_sampling_is_paired_reproducible_and_does_not_change_global_rng(saved_report):
    before = random.getstate()
    selected = pipeline_reports.select_report_songs(saved_report)
    assert len(selected) == 3
    assert len({s["song_id"] for s in selected}) == 3
    shuffled = {**saved_report, "per_song": saved_report["per_song"][::-1]}
    assert [s["song_id"] for s in selected] == [
        s["song_id"] for s in pipeline_reports.select_report_songs(shuffled)
    ]
    assert random.getstate() == before
    assert len(pipeline_reports.select_report_songs(saved_report, count=20)) == 5


def test_role_bridge_report_retains_predicted_start_count_source(saved_report):
    predictor = saved_report["methods"]["bridge"]["predictor"]
    predictor["slot_count_source"] = "predicted_syllable_starts"
    text = pipeline_reports.detailed_markdown(saved_report, saved_report["per_song"][0], "bridge")
    assert "predicted syllable starts with melisma continuations" in text
    assert "one prosody slot per note" not in text


def test_exports_infer_style_reports_lyrics_reference_tail_and_midi_comparison(
    tmp_path, saved_report
):
    before = copy.deepcopy(saved_report)
    folder = tmp_path / "details"
    metadata = pipeline_reports.export_readable_reports(saved_report, folder)
    assert len(metadata["song_ids"]) == 3
    assert saved_report == before
    for song_id in metadata["song_ids"]:
        song_dir = folder / song_id
        assert (song_dir / "reference.lyrics.txt").read_text() == (
            "word word word word\nreference only\n"
        )
        for name in ("actual", "bridge"):
            assert (song_dir / f"{name}.lyrics.txt").read_text() == "word word word word\n"
            text = (song_dir / f"{name}.md").read_text()
            assert "## Pipeline comparison" in text
            assert "Input / reference" in text and "Generated / reference" in text
            assert "## Generated lyrics" in text
            assert "word / æ" in text
            assert "<strong,short>" in text
            assert "Reference conditional perplexity: **3**" in text
            assert "## Phrase 2" in text and "Reference-only phrase" in text
            assert "Ground-truth lyrics: reference only" in text
        assert "MIDI notes and beat-based comparison" in (song_dir / "bridge.md").read_text()
    index = (folder / "index.md").read_text()
    assert "random sample, seed `1234`" in index
    assert index.count("[Generated lyrics]") == 6


def test_explicit_selection_failures_and_ppl_errors_are_readable(tmp_path, saved_report):
    row = saved_report["per_song"][0]
    row["methods"]["actual"] = {"error": "source too long", "stage": "generation"}
    row["methods"]["bridge"]["generation"]["hit_token_limit"] = True
    row["methods"]["bridge"]["reference_perplexity"] = {
        "perplexity": None,
        "error": "target too long",
    }
    folder = tmp_path / "selected"
    meta = pipeline_reports.export_readable_reports(
        saved_report,
        folder,
        songs=["song_0", "song_0", "song_4"],
    )
    assert meta["song_ids"] == ["song_0", "song_4"]
    assert meta["selection"] == "explicit"
    assert not (folder / "song_0" / "actual.lyrics.txt").exists()
    assert "source too long" in (folder / "song_0" / "actual.md").read_text()
    text = (folder / "song_0" / "bridge.md").read_text()
    assert "target too long" in text
    assert "Generation reached its token budget" in text


def test_failures_excluded_from_random_pool_but_available_explicitly(tmp_path, saved_report):
    for name in ("actual", "bridge"):
        saved_report["per_song"][0]["methods"][name] = {"error": "bad MIDI", "stage": "template"}
    assert "song_0" not in {
        s["song_id"] for s in pipeline_reports.select_report_songs(saved_report, count=10)
    }
    assert pipeline_reports.export_readable_reports(
        saved_report,
        tmp_path / "failed",
        songs=["song_0"],
    )["song_ids"] == ["song_0"]


def test_validation_disabling_and_existing_directory_protection(tmp_path, saved_report):
    folder = tmp_path / "details"
    assert pipeline_reports.export_readable_reports(saved_report, folder, count=0) is None
    assert not folder.exists()
    with pytest.raises(ValueError, match="nonnegative"):
        pipeline_reports.export_readable_reports(saved_report, folder, count=-1)
    with pytest.raises(ValueError, match="Unknown report songs"):
        pipeline_reports.export_readable_reports(saved_report, folder, songs=["absent"])
    assert not folder.exists()
    saved_report["per_song"][0]["song_id"] = "../escape"
    with pytest.raises(ValueError, match="Unsafe report filename"):
        pipeline_reports.export_readable_reports(saved_report, folder, songs=["../escape"])
    assert not folder.exists()
    folder.mkdir()
    sentinel = folder / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        pipeline_reports.export_readable_reports(saved_report, folder, songs=["song_1"])
    assert sentinel.read_text() == "keep"


def test_report_only_cli_never_loads_models_or_reextracts_ipa(tmp_path, saved_report, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Report-only export must not load models, read MIDI, or pronounce text"
        )

    monkeypatch.setattr(pipeline_eval, "TemplateDecoder", forbidden)
    monkeypatch.setattr(pipeline_eval, "BridgePredictor", forbidden)
    monkeypatch.setattr(pipeline_eval, "discover_songs", forbidden)
    monkeypatch.setattr(pipeline_eval, "read_references", forbidden)
    source = tmp_path / "saved.json"
    source.write_text(json.dumps(saved_report))
    before = source.read_bytes()
    result = pipeline_eval.main(["--from-report", str(source)])
    assert len(result["readable_reports"]["song_ids"]) == 3
    assert source.read_bytes() == before
    assert source.with_suffix(".details").joinpath("index.md").is_file()
    result = pipeline_eval.main(
        [
            "--from-report",
            str(source),
            "--report-songs",
            "song_0",
            "song_2",
            "--report-dir",
            str(tmp_path / "chosen"),
        ]
    )
    assert result["readable_reports"]["song_ids"] == ["song_0", "song_2"]


def test_cli_rejects_unknown_songs_and_negative_counts_before_any_output(tmp_path, saved_report):
    source = tmp_path / "saved.json"
    source.write_text(json.dumps(saved_report))
    for options in (["--report-songs", "absent"], ["--report-count", "-1"]):
        with pytest.raises(SystemExit):
            pipeline_eval.main(["--from-report", str(source), *options])
    assert not source.with_suffix(".details").exists()
