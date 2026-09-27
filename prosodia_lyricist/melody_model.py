"""Direct note conditioning with the unchanged explainable four-stream decoder."""

import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration, GenerationConfig

from .generation import generate_templates
from .melody_encoder.encoding import MELODY_REPRESENTATION
from .melody_encoder.modeling import MelodyTransformerEncoder
from .model import ProsodyBart


class MelodyBart(ProsodyBart):
    def __init__(
        self,
        bart,
        *,
        max_syllables,
        melody_config,
        dropout=0.1,
        loss_weights=None,
        provenance=None,
        source_config=None,
    ):
        super().__init__(
            bart, max_syllables=max_syllables, dropout=dropout, loss_weights=loss_weights
        )
        self.melody_config = {**melody_config, "projection_dim": None}
        self.melody_encoder = MelodyTransformerEncoder(**self.melody_config)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.melody_encoder.d_model),
            nn.Linear(self.melody_encoder.d_model, bart.config.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bart.config.d_model, bart.config.d_model),
        )
        self.provenance = provenance or {}
        self.source_config = source_config or {"lines_per_window": 1, "include_title": True}
        self.set_trainable(melody=False, decoder=True)

    @classmethod
    def from_template(cls, template, *, loss_weights=None, **kwargs):
        if type(template) is not ProsodyBart:
            raise ValueError("Initialization requires a four-stream template-decoder checkpoint")
        model = cls(
            template.bart,
            max_syllables=template.max_syllables,
            dropout=template.dropout_probability,
            loss_weights=template.loss_weights if loss_weights is None else loss_weights,
            **kwargs,
        )
        # Copy every pretrained compound embedding, projection, and auxiliary head.
        state = model.state_dict()
        state.update(template.state_dict())
        model.load_state_dict(state, strict=True)
        return model

    def set_trainable(self, *, melody, decoder):
        self.melody_frozen, self.decoder_frozen = not melody, not decoder
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                melody
                if name.startswith("melody_encoder.")
                else True
                if name.startswith("adapter.")
                else decoder
            )
        # Source template projection is retained for checkpoint fidelity but unused.
        self.projection.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "melody_frozen", False):
            self.melody_encoder.eval()
        if getattr(self, "decoder_frozen", False):
            for name, module in self.named_children():
                if name not in ("adapter", "melody_encoder"):
                    module.eval()
        return self

    def embed_melody(
        self,
        input_ids,
        attention_mask,
        melody_features,
        melody_attention_mask,
        note_positions,
    ):
        """Encode local windows, then scatter all notes into the shared song source."""
        mask = melody_attention_mask.bool()
        if mask.shape != melody_features.shape[:2] or note_positions.shape != mask.shape:
            raise ValueError("Window features, masks and note positions must agree")
        if not mask.any(dim=1).all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("Melody windows must be nonempty and right-padded")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("Source IDs and attention mask must agree")
        if input_ids.shape[1] > self.bart.config.max_position_embeddings:
            raise ValueError("Song source exceeds BART positions; no truncation")
        positions = note_positions[mask]
        if (positions < 0).any() or (positions >= input_ids.numel()).any():
            raise ValueError("Note position outside the source")
        if positions.unique().numel() != positions.numel():
            raise ValueError("Each note must occupy its own source position")
        if not attention_mask.reshape(-1)[positions].bool().all():
            raise ValueError("Notes cannot occupy source padding")
        with torch.no_grad() if self.melody_frozen else nullcontext():
            notes = self.melody_encoder.encode(
                melody_features.masked_fill(~mask.unsqueeze(-1), 0),
                mask,
                normalize=False,
                project=False,
            ).note_embeddings
        notes = self.adapter(notes)[mask]
        source = self.bart.get_encoder().embed_tokens(input_ids)
        return (
            source.reshape(-1, source.shape[-1])
            .index_copy(0, positions, notes.to(source.dtype))
            .reshape_as(source)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        melody_features,
        melody_attention_mask,
        note_positions,
        labels,
        syllable_labels,
        stress_labels,
        length_labels,
    ):
        source = self.embed_melody(
            input_ids,
            attention_mask,
            melody_features,
            melody_attention_mask,
            note_positions,
        )
        return self.forward_embedded(
            source,
            attention_mask,
            labels,
            syllable_labels,
            stress_labels,
            length_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        melody_features,
        melody_attention_mask,
        note_positions,
        *,
        tokenizer,
        **kwargs,
    ):
        source = self.embed_melody(
            input_ids,
            attention_mask,
            melody_features,
            melody_attention_mask,
            note_positions,
        )
        return generate_templates(
            self,
            tokenizer,
            input_ids,
            None,
            attention_mask,
            source_embeds=source,
            **kwargs,
        )

    def save(self, directory, tokenizer):
        directory = Path(directory)
        super().save(directory, tokenizer)
        metadata = {
            "format_version": 1,
            "melody_representation": MELODY_REPRESENTATION,
            "melody_config": self.melody_config,
            "provenance": self.provenance,
            "source_config": self.source_config,
        }
        (directory / "melody.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "melody.json").read_text())
        if meta.pop("format_version") != 1 or meta.pop("melody_representation") != (
            MELODY_REPRESENTATION
        ):
            raise ValueError("Incompatible direct-melody checkpoint")
        prosody = json.loads((directory / "prosody.json").read_text())
        if (
            prosody.pop("format_version") != 2
            or prosody.pop("word_boundary") != "<word_end>"
            or prosody.pop("prosody_rules") != "ipa_binary_stress_diphthong_length_v2"
        ):
            raise ValueError("Incompatible four-stream checkpoint")
        model = cls(
            BartForConditionalGeneration(
                BartConfig.from_pretrained(directory, local_files_only=True)
            ),
            **meta,
            **prosody,
        )
        model.bart.generation_config = GenerationConfig.from_pretrained(
            directory, local_files_only=True
        )
        model.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True), strict=True
        )
        return model
