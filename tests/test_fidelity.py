import copy
import json
from types import SimpleNamespace

import pytest
import torch

from prosodia_lyricist import ipa
from prosodia_lyricist.dali import extract_lines
from prosodia_lyricist.prepare import prepare
from prosodia_lyricist.train import build_scheduler


def fake_backend(monkeypatch):
    words = [
        SimpleNamespace(
            txt="hello", wordtype=SimpleNamespace(form=SimpleNamespace(syllables=["hə", "'ləʊ"]))
        ),
        SimpleNamespace(
            txt="world", wordtype=SimpleNamespace(form=SimpleNamespace(syllables=["`wɜːld"]))
        ),
    ]
    monkeypatch.setattr(
        ipa,
        "backend",
        lambda: SimpleNamespace(Text=lambda *a, **kw: SimpleNamespace(wordtokens=words)),
    )


def test_ipa_rules():
    assert ipa.syllable_features("'ɑː")["stress"] == "strong"
    assert ipa.syllable_features("`ə")["stress"] == "strong"
    assert ipa.syllable_features("ə")["stress"] == "weak"
    assert ipa.syllable_features("'ɑː")["length"] == "long"
    assert ipa.syllable_features("'aɪ")["length"] == "long"


def test_ipa_template_independent_of_sung_count(monkeypatch, annotation):
    fake_backend(monkeypatch)
    annot = annotation["annotations"]["annot"]
    # DALI now has one sung syllable for hello, versus two in IPA.
    annot["notes"][:3] = [{"text": "hello", "time": [0, 2], "index": 0}]
    records, rejects = extract_lines(annotation["info"], annot, stress_source="ipa")
    assert not rejects
    record = records[0]
    assert len(record["syllables"]) == 3
    assert len(record["sung_syllables"]) == 2
    assert [s["length"] for s in record["syllables"]] == ["short", "long", "long"]
    assert [s["stress"] for s in record["syllables"]] == ["weak", "strong", "strong"]
    assert [w["syllable_count"] for w in record["words"]] == [2, 1]


def test_real_modern_parser():
    pytest.importorskip("prosodic")
    words = ipa.parse_words("hello world")
    assert sum(len(w["syllables"]) for w in words) == 3
    assert all(s["ipa"] for w in words for s in w["syllables"])


def test_schedule_matches_literal_original():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1)
    schedule = build_scheduler(optimizer, {"schedule": "xai_original", "warmup_steps": 4}, 20)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        schedule.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates == [0, 0.25, 0.5, 0.75, 0, 0, 0]


def test_constant_schedule_without_warmup():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1)
    build_scheduler(optimizer, {"schedule": "constant_after_warmup", "warmup_steps": 0}, 20)
    assert optimizer.param_groups[0]["lr"] == 1


def test_id_selection_and_language_provenance(tmp_path, annotation):
    raw = tmp_path / "raw"
    raw.mkdir()
    for song_id, language in (
        ("wanted", "english"),
        ("french", "french"),
        ("unknown", ""),
        ("outside", "english"),
    ):
        row = copy.deepcopy(annotation)
        row["info"]["id"] = song_id
        row["info"]["metadata"]["language"] = language
        (raw / f"unrelated-name-{song_id}.json").write_text(json.dumps(row))
    ids = tmp_path / "ids.txt"
    ids.write_text("wanted\nfrench\nunknown\nmissing\nwanted\n\n")
    config = {
        "data": {
            "dali_dir": str(raw),
            "prepared_dir": str(tmp_path / "out"),
            "song_ids_file": str(ids),
            "language": "english",
            "stress_source": "lexical",
            "min_ncc": 0,
            "max_syllables": 40,
            "seed": 1234,
            "valid_fraction": 0.1,
            "test_fraction": 0.1,
        }
    }
    manifest = prepare(config)
    assert set(manifest["songs"]) == {"wanted"}
    assert manifest["selection"]["missing_ids"] == ["missing"]
    assert manifest["selection"]["unprepared_ids"] == ["french", "missing", "unknown"]
    assert manifest["counts"]["songs_filtered_language"] == 2
    assert manifest["selection"]["sha256"]
