import copy
import json
from types import SimpleNamespace

import pytest
import torch

from prosodia_lyricist import ipa
from prosodia_lyricist.dali import extract_lines
from prosodia_lyricist.data import ProsodyCollator
from prosodia_lyricist.features import encode_example
from prosodia_lyricist.model import ProsodyBart
from prosodia_lyricist.prepare import prepare
from prosodia_lyricist.train import build_scheduler, run_epoch


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
    assert ipa.syllable_features("`ə")["stress"] == "substrong"
    assert ipa.syllable_features("ə")["stress"] == "weak"
    assert ipa.syllable_features("'ɑː")["length"] == "long"
    assert ipa.syllable_features("'aɪ")["length"] == "short"


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
    assert [s["length"] for s in record["syllables"]] == ["short", "short", "long"]
    assert [s["stress"] for s in record["syllables"]] == ["weak", "strong", "substrong"]
    assert [w["syllable_count"] for w in record["words"]] == [2, 1]


def test_real_modern_parser():
    pytest.importorskip("prosodic")
    words = ipa.parse_words("hello world")
    assert sum(len(w["syllables"]) for w in words) == 3
    assert all(s["ipa"] for w in words for s in w["syllables"])


def test_auxiliary_losses_gradients_padding_and_reload(tmp_path, tokenizer, tiny_model, record):
    record["words"] = [
        {"text": "hello", "syllable_count": 2},
        {"text": "world", "syllable_count": 1},
    ]
    example = encode_example(record, tokenizer, 8, scaffold=True)
    assert example["labels"] == encode_example(record, tokenizer, 8)["labels"]
    assert example["syllable_labels"] == [-100, 2, 1, -100, -100]
    assert example["remainder_labels"] == [-100, 1, 0, -100, -100]
    short = encode_example(
        {**record, "words": record["words"][:1], "syllables": record["syllables"][:2]},
        tokenizer,
        8,
        scaffold=True,
    )
    batch = ProsodyCollator(tokenizer.pad_token_id)([example, short])
    assert batch["remainder_labels"][1, -1] == -100
    model = ProsodyBart(
        tiny_model.bart,
        max_syllables=8,
        dropout=0,
        loss_weights={"word": 1, "syllable": 0.3, "remainder": 0.4, "sentence": 0.2},
    )
    output = model(**batch)
    assert torch.allclose(
        output.loss, sum(model.loss_weights[k] * v for k, v in output.loss_components.items())
    )
    output.loss.backward()
    for head in model.auxiliary_heads.values():
        assert head.weight.grad.abs().sum() > 0
    assert model.projection.weight.grad.abs().sum() > 0
    model.eval().save(tmp_path, tokenizer)
    restored = ProsodyBart.load(tmp_path).eval()
    assert restored.loss_weights == model.loss_weights
    assert torch.equal(model(**batch).loss, restored(**batch).loss)
    with pytest.raises(ValueError, match="scaffold labels"):
        model(**{k: v for k, v in batch.items() if k != "remainder_labels"})


def test_text_only_creates_no_heads(tiny_model):
    assert not tiny_model.auxiliary_heads


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


def test_original_batch_mean():
    class Model:
        def train(self, mode):
            pass

        def __call__(self, labels):
            return SimpleNamespace(loss=labels[0, 0].float())

    metrics = run_epoch(
        Model(),
        [{"labels": torch.tensor([[2]])}, {"labels": torch.tensor([[4, 4, 4]])}],
        "cpu",
        loss_aggregation="batches",
    )
    assert metrics["loss"] == 3


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
