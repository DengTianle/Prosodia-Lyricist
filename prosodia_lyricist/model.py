"""Explainable Prosody BART: compound decoder inputs and four cross-entropy losses."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration, GenerationConfig

FORMAT_VERSION = 2
LOSS_NAMES = ("lyrics", "syllables", "stresses", "lengths")


class ProsodyBart(nn.Module):
    def __init__(self, bart, *, max_syllables, dropout=0.1, loss_weights=None):
        super().__init__()
        self.bart = bart
        self.max_syllables = max_syllables
        self.dropout_probability = dropout
        size = bart.config.d_model
        self.length_embedding = nn.Embedding(3, size, padding_idx=0)
        self.syllable_embedding = nn.Embedding(max_syllables + 1, size, padding_idx=0)
        self.stress_embedding = nn.Embedding(3, size, padding_idx=0)
        self.projection = nn.Linear(2 * size, size)
        self.decoder_projection = nn.Linear(4 * size, size)
        self.dropout = nn.Dropout(dropout)
        self.prosody_heads = nn.ModuleDict(
            {
                "syllables": nn.Linear(size, max_syllables + 1),
                "stresses": nn.Linear(size, 3),
                "lengths": nn.Linear(size, 3),
            }
        )
        for embedding in (self.length_embedding, self.syllable_embedding, self.stress_embedding):
            nn.init.normal_(embedding.weight, mean=0, std=size**-0.5)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.loss_weights = dict.fromkeys(LOSS_NAMES, 1.0)
        if loss_weights:
            if set(loss_weights) - self.loss_weights.keys():
                raise ValueError(f"Explainable loss names must be {LOSS_NAMES}")
            self.loss_weights.update(loss_weights)
        if any(not 0 <= v < float("inf") for v in self.loss_weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not any(self.loss_weights.values()):
            raise ValueError("At least one loss must be enabled")

    def embed_source(self, input_ids, length_ids):
        words = self.bart.get_encoder().embed_tokens(input_ids)
        return self.dropout(
            self.projection(torch.cat((words, self.length_embedding(length_ids)), -1))
        )

    def embed_target(self, input_ids, syllable_ids, stress_ids, length_ids):
        return self.dropout(
            self.decoder_projection(
                torch.cat(
                    (
                        self.bart.get_decoder().embed_tokens(input_ids),
                        self.syllable_embedding(syllable_ids),
                        self.stress_embedding(stress_ids),
                        self.length_embedding(length_ids),
                    ),
                    dim=-1,
                )
            )
        )

    @staticmethod
    def shift_features(labels):
        shifted = torch.zeros_like(labels)
        shifted[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), 0)
        return shifted

    def forward(
        self,
        input_ids,
        length_ids,
        attention_mask,
        labels,
        syllable_labels,
        stress_labels,
        length_labels,
    ):
        targets = (syllable_labels, stress_labels, length_labels)
        for target in targets:
            if target.shape != labels.shape or not torch.equal(target.eq(-100), labels.eq(-100)):
                raise ValueError("All four target streams must have aligned padding and shapes")
        decoder_ids = self.bart.prepare_decoder_input_ids_from_labels(labels)
        decoder_features = [self.shift_features(target) for target in targets]
        decoder_mask = torch.ones_like(labels)
        decoder_mask[:, 1:] = labels[:, :-1].ne(-100)
        output = self.bart(
            inputs_embeds=self.embed_source(input_ids, length_ids),
            attention_mask=attention_mask,
            decoder_inputs_embeds=self.embed_target(decoder_ids, *decoder_features),
            decoder_attention_mask=decoder_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
        )
        logits = {
            name: head(output.decoder_hidden_states[-1])
            for name, head in self.prosody_heads.items()
        }
        losses = {"lyrics": output.loss}
        for (name, values), target in zip(logits.items(), targets, strict=True):
            losses[name] = nn.functional.cross_entropy(
                values.reshape(-1, values.shape[-1]), target.reshape(-1), ignore_index=-100
            )
        output.loss = sum(self.loss_weights[name] * loss for name, loss in losses.items())
        output["loss_components"] = losses
        output["prosody_logits"] = logits
        return output

    @torch.no_grad()
    def generate(self, input_ids, length_ids, attention_mask, *, tokenizer, **kwargs):
        from .generation import generate_templates

        return generate_templates(self, tokenizer, input_ids, length_ids, attention_mask, **kwargs)

    def save(self, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bart.config.save_pretrained(directory)
        self.bart.generation_config.save_pretrained(directory)
        tokenizer.save_pretrained(directory / "tokenizer")
        metadata = {
            "format_version": FORMAT_VERSION,
            "max_syllables": self.max_syllables,
            "dropout": self.dropout_probability,
            "loss_weights": self.loss_weights,
            "word_boundary": "<word_end>",
            "prosody_rules": "ipa_binary_stress_diphthong_length_v2",
        }
        (directory / "prosody.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary = directory / "weights.pt.tmp"
        torch.save(self.state_dict(), temporary)
        temporary.replace(directory / "weights.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "prosody.json").read_text(encoding="utf-8"))
        if "format_version" not in metadata:
            from .legacy_model import ProsodyBart as LegacyProsodyBart

            return LegacyProsodyBart.load(directory)
        if metadata.pop("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported explainable checkpoint version")
        if metadata.pop("word_boundary") != "<word_end>" or metadata.pop("prosody_rules") != (
            "ipa_binary_stress_diphthong_length_v2"
        ):
            raise ValueError("Checkpoint has incompatible word boundary or IPA rules")
        config = BartConfig.from_pretrained(directory, local_files_only=True)
        model = cls(BartForConditionalGeneration(config), **metadata)
        model.bart.generation_config = GenerationConfig.from_pretrained(
            directory, local_files_only=True
        )
        model.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True), strict=True
        )
        return model
