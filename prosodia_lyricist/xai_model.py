"""Load the original XAI-Lyricist checkpoint and train its single lyric stream.

python -m prosodia_lyricist.xai_model --checkpoint checkpoints/bestM2LCkpt.pt \
    --output-dir checkpoints/xai-original --local-files-only
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration, GenerationConfig

from .melody_checkpoint import file_sha256


class XAILyricist(nn.Module):
    decoder_mode = "xai"

    def __init__(
        self, bart, *, max_syllables=40, remainder_size=42, provenance=None, loss_weights=None
    ):
        super().__init__()
        if not 1 <= max_syllables < remainder_size:
            raise ValueError("Syllable counts must fit the source remainder vocabulary")
        self.bart = bart
        self.max_syllables, self.remainder_size = max_syllables, remainder_size
        self.provenance = provenance or {}
        self.loss_weights = loss_weights or {"lyrics": 1.0}
        if set(self.loss_weights) != {"lyrics"} or not 0 < self.loss_weights["lyrics"] < math.inf:
            raise ValueError("The XAI checkpoint has only a positive lyrics loss")
        size = bart.config.d_model
        self.length_embedding = nn.Embedding(3, size, padding_idx=0)
        self.remainder_embedding = nn.Embedding(remainder_size, size, padding_idx=0)
        self.projection = nn.Linear(3 * size, size)
        # The reference constructs embedding dropout but never calls it.
        self.dropout = nn.Identity()
        # Reference inference starts with <s>. Do not inherit summarization defaults.
        self.bart.config.decoder_start_token_id = self.bart.config.bos_token_id
        self.bart.generation_config = GenerationConfig(
            bos_token_id=bart.config.bos_token_id,
            # Reference infer checks tokenizer.encode('</s>'), containing BOS and EOS.
            eos_token_id=[bart.config.bos_token_id, bart.config.eos_token_id],
            pad_token_id=bart.config.pad_token_id,
            decoder_start_token_id=bart.config.bos_token_id,
            num_beams=1,
            no_repeat_ngram_size=0,
            repetition_penalty=1.0,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
        )

    def embed_source(self, input_ids, length_ids, remainder_ids):
        if input_ids.shape != length_ids.shape or input_ids.shape != remainder_ids.shape:
            raise ValueError("XAI source meter, length, and remainder IDs must align")
        return self.projection(
            torch.cat(
                (
                    self.bart.get_encoder().embed_tokens(input_ids),
                    self.length_embedding(length_ids),
                    self.remainder_embedding(remainder_ids),
                ),
                dim=-1,
            )
        )

    def forward(
        self, input_ids, length_ids, remainder_ids, attention_mask, labels, *, inputs_embeds=None
    ):
        decoder_mask = torch.ones_like(labels)
        decoder_mask[:, 1:] = labels[:, :-1].ne(-100)
        output = self.bart(
            inputs_embeds=(
                self.embed_source(input_ids, length_ids, remainder_ids)
                if inputs_embeds is None
                else inputs_embeds
            ),
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_mask,
            use_cache=False,
        )
        output["loss_components"] = {"lyrics": output.loss}
        output.loss = self.loss_weights["lyrics"] * output.loss
        return output

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        length_ids,
        remainder_ids,
        attention_mask,
        *,
        tokenizer,
        inputs_embeds=None,
        max_new_tokens=1023,
        **kwargs,
    ):
        if self.training:
            raise ValueError("Call model.eval() before generation")
        if not 1 <= max_new_tokens < self.bart.config.max_position_embeddings:
            raise ValueError("Generation budget must fit the checkpoint position limit")
        tokens = self.bart.generate(
            inputs_embeds=(
                self.embed_source(input_ids, length_ids, remainder_ids)
                if inputs_embeds is None
                else inputs_embeds
            ),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        explanations = []
        for row in tokens:
            text = tokenizer.decode(row, skip_special_tokens=True)
            terminals = row[1:][
                (row[1:] == self.bart.config.eos_token_id)
                | (row[1:] == self.bart.config.bos_token_id)
            ]
            completed = bool(len(terminals))
            hit_limit = not completed and len(row) - 1 >= max_new_tokens
            explanations.append(
                dict(
                    decoder_mode="xai_single_stream",
                    text=text,
                    lines=[s.strip() for s in text.split(".") if s.strip()],
                    completed=completed,
                    hit_token_limit=hit_limit,
                    stop_reason=(
                        "bos"
                        if completed and terminals[0] == self.bart.config.bos_token_id
                        else "eos"
                        if completed
                        else "token_limit"
                        if hit_limit
                        else "custom_stop"
                    ),
                )
            )
        return SimpleNamespace(sequences=tokens, explanations=explanations)

    def save(self, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bart.config.save_pretrained(directory)
        self.bart.generation_config.save_pretrained(directory)
        tokenizer.save_pretrained(directory / "tokenizer")
        metadata = dict(
            format_version=1,
            architecture="xai_single_stream",
            max_syllables=self.max_syllables,
            remainder_size=self.remainder_size,
            loss_weights=self.loss_weights,
            provenance=self.provenance,
        )
        (directory / "xai.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary = directory / "weights.pt.tmp"
        torch.save(self.state_dict(), temporary)
        temporary.replace(directory / "weights.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "xai.json").read_text(encoding="utf-8"))
        if (
            metadata.pop("format_version") != 1
            or metadata.pop("architecture") != "xai_single_stream"
        ):
            raise ValueError("Unsupported XAI checkpoint format")
        config = BartConfig.from_pretrained(directory, local_files_only=True)
        model = cls(BartForConditionalGeneration(config), **metadata)
        model.bart.generation_config = GenerationConfig.from_pretrained(
            directory, local_files_only=True
        )
        model.bart.generation_config.eos_token_id = [config.bos_token_id, config.eos_token_id]
        model.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True), strict=True
        )
        return model

    @classmethod
    def from_original(cls, checkpoint, *, config=None, max_syllables=40):
        """Map by exact names and reject missing, extra, or inconsistent tied tensors."""
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(v, torch.Tensor) for v in state.values()
        ):
            raise ValueError("Expected an original XAI-Lyricist state_dict")
        shared = state["model.model.shared.weight"]
        vocab, size = shared.shape
        config_source = "supplied"
        if config is None:
            if (vocab, size) != (50322, 768):
                raise ValueError("Supply the original BART config for a nonstandard XAI checkpoint")
            # These head counts / activations are not identifiable from tensor shapes.
            # The reference loads bart-base; its n_head/n_layers arguments are unused.
            config = BartConfig(
                vocab_size=vocab,
                d_model=size,
                encoder_layers=6,
                decoder_layers=6,
                encoder_attention_heads=12,
                decoder_attention_heads=12,
                encoder_ffn_dim=3072,
                decoder_ffn_dim=3072,
                max_position_embeddings=1024,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.1,
                scale_embedding=False,
            )
            config_source = "bart-base inference; original custom config unavailable"
        elif not isinstance(config, BartConfig):
            config = BartConfig.from_pretrained(config, local_files_only=True)
        if config.vocab_size != vocab or config.d_model != size:
            raise ValueError("BART config differs from the original checkpoint dimensions")
        aliases = (
            "model.model.encoder.embed_tokens.weight",
            "model.model.decoder.embed_tokens.weight",
            "model.lm_head.weight",
            "src_word_emb.weight",
            "mel_embed.meter_emb.weight",
            "tgt_word_emb.weight",
            "lyr_embed.tgt_word_emb.weight",
        )
        for key in aliases:
            if not torch.equal(state[key], shared):
                raise ValueError(f"Original checkpoint has inconsistent shared embedding: {key}")
        ignored_aliases = set(aliases[3:])
        mapping = {
            "mel_embed.leng_emb.weight": "length_embedding.weight",
            "mel_embed.rem_emb.weight": "remainder_embedding.weight",
            "mel_embed.token_emb_proj.weight": "projection.weight",
            "mel_embed.token_emb_proj.bias": "projection.bias",
        }
        mapped = {}
        for key, value in state.items():
            if key in ignored_aliases or key == "pos_enc.inv_freq":
                continue
            if key.startswith("model."):
                target = "bart." + key.removeprefix("model.")
            elif key in mapping:
                target = mapping[key]
            else:
                raise ValueError(f"Unexpected original checkpoint tensor: {key}")
            mapped[target] = value
        expected_frequency = 1 / (10000 ** (torch.arange(0, size, 2).float() / size))
        if "pos_enc.inv_freq" not in state or not torch.allclose(
            state["pos_enc.inv_freq"], expected_frequency
        ):
            raise ValueError("Unexpected reference positional-encoding buffer")
        model = cls(
            BartForConditionalGeneration(config),
            max_syllables=max_syllables,
            remainder_size=state["mel_embed.rem_emb.weight"].shape[0],
            provenance=dict(
                original_checkpoint=str(Path(checkpoint).resolve()),
                original_sha256=file_sha256(checkpoint),
                original_tensor_count=len(state),
                output_streams=["lyrics"],
                config_source=config_source,
                unused_buffer="pos_enc.inv_freq (never used by reference forward/infer)",
                training_shift="causal BOS prefix; reference forward has unshifted target inputs",
            ),
        )
        model.load_state_dict(mapped, strict=True)
        return model


def load_template(
    checkpoint,
    *,
    tokenizer_dir=None,
    bart_config=None,
    cache_dir=None,
    local_files_only=False,
    max_syllables=40,
):
    from .xai_features import load_tokenizer, validate_tokenizer

    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        if tokenizer_dir is not None or bart_config is not None:
            raise ValueError("Tokenizer/config overrides require an original XAI checkpoint file")
        model = XAILyricist.load(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    else:
        model = XAILyricist.from_original(
            checkpoint, config=bart_config, max_syllables=max_syllables
        )
        tokenizer = load_tokenizer(
            tokenizer_dir, cache_dir=cache_dir, local_files_only=local_files_only
        )
        model.provenance["tokenizer"] = tokenizer.init_kwargs["xai_tokenizer_provenance"]
    validate_tokenizer(tokenizer, model)
    return model, tokenizer


def main():
    from .prepare import SCHEMA_VERSION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", help="Author's original source tokenizer, if available")
    parser.add_argument("--bart-config", help="Author's original model config, if available")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    model, tokenizer = load_template(
        args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        bart_config=args.bart_config,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model.save(args.output_dir, tokenizer)
    run = dict(
        decoder_mode="xai",
        data_schema_version=SCHEMA_VERSION,
        provenance=model.provenance,
        config=dict(
            data=dict(stress_source="ipa"),
            model=dict(
                max_source_length=model.bart.config.max_position_embeddings,
                max_target_length=model.bart.config.max_position_embeddings,
            ),
        ),
    )
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(dict(checkpoint=str(args.output_dir), **model.provenance), indent=2))


if __name__ == "__main__":
    main()
