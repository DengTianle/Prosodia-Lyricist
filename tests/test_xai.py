"""Checkpoint import fidelity, causal single-stream training, and source semantics."""

import copy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from prosodia_lyricist.data import ProsodyCollator
from prosodia_lyricist.model import ProsodyBart
from prosodia_lyricist.xai_features import encode_example, encode_targets, load_tokenizer
from prosodia_lyricist.xai_model import XAILyricist


@pytest.fixture
def xai_model(tiny_model):
    return XAILyricist(tiny_model.bart, max_syllables=8)


def original_state(model):
    state = {"model." + k: v for k, v in model.bart.state_dict().items()}
    shared = state["model.model.shared.weight"]
    for key in (
        "src_word_emb.weight",
        "mel_embed.meter_emb.weight",
        "tgt_word_emb.weight",
        "lyr_embed.tgt_word_emb.weight",
    ):
        state[key] = shared
    state.update(
        {
            "mel_embed.leng_emb.weight": model.length_embedding.weight,
            "mel_embed.rem_emb.weight": model.remainder_embedding.weight,
            "mel_embed.token_emb_proj.weight": model.projection.weight,
            "mel_embed.token_emb_proj.bias": model.projection.bias,
            "pos_enc.inv_freq": 1
            / (
                10000
                ** (
                    torch.arange(0, model.bart.config.d_model, 2).float()
                    / model.bart.config.d_model
                )
            ),
        }
    )
    return state


def test_original_checkpoint_maps_every_tensor_and_rejects_extras(xai_model, tmp_path):
    path = tmp_path / "original.pt"
    state = original_state(xai_model)
    torch.save(state, path)
    imported = XAILyricist.from_original(path, config=xai_model.bart.config, max_syllables=8)
    for name, value in xai_model.state_dict().items():
        assert torch.equal(value, imported.state_dict()[name])
    state["unknown_head.weight"] = torch.ones(3, 16)
    torch.save(state, path)
    with pytest.raises(ValueError, match="Unexpected.*tensor"):
        XAILyricist.from_original(path, config=xai_model.bart.config, max_syllables=8)
    del state["unknown_head.weight"]
    state["src_word_emb.weight"] = state["src_word_emb.weight"] + 1
    torch.save(state, path)
    with pytest.raises(ValueError, match="inconsistent shared"):
        XAILyricist.from_original(path, config=xai_model.bart.config, max_syllables=8)
    del state["src_word_emb.weight"]
    state["src_word_emb.weight"] = state["model.model.shared.weight"]
    del state["mel_embed.leng_emb.weight"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        XAILyricist.from_original(path, config=xai_model.bart.config, max_syllables=8)


def test_single_stream_uses_previous_lyric_only_and_padding_is_ignored(
    xai_model, record, tokenizer
):
    model = xai_model.eval()
    encoded = encode_example(record, tokenizer, 8)
    assert set(encoded) == {"input_ids", "length_ids", "remainder_ids", "attention_mask", "labels"}
    assert encoded["labels"][0] != tokenizer.bos_token_id
    assert [n for n in encoded["remainder_ids"] if n] == [3, 2, 1]
    batch = ProsodyCollator(tokenizer.pad_token_id)([encoded])
    captured = {}

    def capture(module, args, kwargs):
        captured["decoder_input_ids"] = kwargs["decoder_input_ids"].clone()

    hook = model.bart.model.register_forward_pre_hook(capture, with_kwargs=True)
    output = model(**batch)
    hook.remove()
    shifted = captured["decoder_input_ids"]
    assert shifted[0, 0] == tokenizer.bos_token_id
    assert torch.equal(shifted[:, 1:], batch["labels"][:, :-1])
    assert set(output.loss_components) == {"lyrics"}
    assert "prosody_logits" not in output
    expected = torch.nn.functional.cross_entropy(
        output.logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=-100
    )
    torch.testing.assert_close(output.loss, expected)
    padded = {
        k: torch.nn.functional.pad(
            v,
            (0, 3),
            value=(-100 if k == "labels" else tokenizer.pad_token_id if k == "input_ids" else 0),
        )
        for k, v in batch.items()
    }
    torch.testing.assert_close(model(**padded).loss, output.loss)
    changed = {**batch, "labels": batch["labels"].clone()}
    changed["labels"][0, 1] = tokenizer.convert_tokens_to_ids("light")
    torch.testing.assert_close(model(**changed).logits[:, :2], output.logits[:, :2])
    output.loss.backward()
    for module in (model.projection, model.length_embedding, model.remainder_embedding):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0


def test_source_projection_matches_reference_without_embedding_dropout(
    xai_model, record, tokenizer
):
    data = encode_example(record, tokenizer, 8)
    batch = ProsodyCollator(tokenizer.pad_token_id)([data])
    xai_model.train()
    expected = torch.nn.functional.linear(
        torch.cat(
            (
                xai_model.bart.get_encoder().embed_tokens(batch["input_ids"]),
                xai_model.length_embedding(batch["length_ids"]),
                xai_model.remainder_embedding(batch["remainder_ids"]),
            ),
            dim=-1,
        ),
        xai_model.projection.weight,
        xai_model.projection.bias,
    )
    actual = xai_model.embed_source(batch["input_ids"], batch["length_ids"], batch["remainder_ids"])
    assert torch.equal(actual, expected)
    assert not hasattr(xai_model, "prosody_heads")
    with pytest.raises(ValueError, match="only.*lyrics loss"):
        XAILyricist(xai_model.bart, loss_weights={"lyrics": 1, "stresses": 1})


def test_native_save_reload_and_plain_generation(xai_model, record, tokenizer, tmp_path):
    model = xai_model.eval()
    model.save(tmp_path, tokenizer)
    restored = ProsodyBart.load(tmp_path).eval()
    assert isinstance(restored, XAILyricist)
    assert restored.bart.generation_config.no_repeat_ngram_size == 0
    assert restored.bart.generation_config.num_beams == 1
    assert restored.bart.generation_config.forced_eos_token_id is None
    saved_tokenizer = AutoTokenizer.from_pretrained(tmp_path / "tokenizer", local_files_only=True)
    assert saved_tokenizer.get_vocab() == tokenizer.get_vocab()
    batch = ProsodyCollator(tokenizer.pad_token_id)([encode_example(record, tokenizer, 8)])
    assert torch.equal(model(**batch).logits, restored(**batch).logits)
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    generated = restored.generate(**inputs, tokenizer=tokenizer, max_new_tokens=5, do_sample=False)
    assert generated.sequences.shape[1] <= 6
    assert generated.sequences[0, 0] == tokenizer.bos_token_id
    assert generated.explanations[0]["decoder_mode"] == "xai_single_stream"


def test_lyric_targets_do_not_require_prosody_labels(record, tokenizer):
    first = encode_targets(record, tokenizer)
    changed = copy.deepcopy(record)
    del changed["syllables"]
    assert encode_targets(changed, tokenizer) == first


def test_generation_stops_on_bos_like_reference_tokenizer_membership(xai_model, record, tokenizer):
    from transformers import LogitsProcessor

    class ForceBOS(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores.fill_(-torch.inf)
            scores[:, tokenizer.bos_token_id] = 0
            return scores

    model = xai_model.eval()
    batch = ProsodyCollator(tokenizer.pad_token_id)([encode_example(record, tokenizer, 8)])
    generated = model.generate(
        **{k: v for k, v in batch.items() if k != "labels"},
        tokenizer=tokenizer,
        logits_processor=[ForceBOS()],
        max_new_tokens=5,
        do_sample=False,
    )
    assert generated.sequences.tolist() == [[tokenizer.bos_token_id, tokenizer.bos_token_id]]
    assert generated.explanations[0]["completed"]
    assert generated.explanations[0]["stop_reason"] == "bos"
    assert not generated.explanations[0]["hit_token_limit"]


def test_recovery_keeps_exact_observed_ids_and_records_assumptions(tokenizer, monkeypatch):
    # Test all 50,322 IDs without a network dependency or arbitrary ID offsets.
    import prosodia_lyricist.xai_features as features

    words = ["<s>", "<pad>", "</s>", "<unk>"] + [f"base_{i}" for i in range(4, 50265)]
    base = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>")
        ),
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    monkeypatch.setattr(features.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: base)
    recovered = load_tokenizer(local_files_only=True)
    assert len(recovered) == 50322
    assert recovered.convert_tokens_to_ids("<strong>") == 50319
    assert recovered.convert_tokens_to_ids("<weak>") == 50321
    assert recovered.convert_tokens_to_ids("<syllable_5>") == 50271
    assert recovered.convert_tokens_to_ids("<syllable_6>") == 50272
    assert recovered.convert_tokens_to_ids("<syllable_7>") == 50273
    assert recovered.convert_ids_to_tokens(50265) == "<xai_reserved_50265>"
    provenance = recovered.init_kwargs["xai_tokenizer_provenance"]
    assert provenance["source"] == "reference_binary_recovery_v1"
    assert provenance["assumptions"]
    monkeypatch.setattr(
        features.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    with pytest.raises(ValueError, match="unmodified"):
        load_tokenizer(local_files_only=True)
