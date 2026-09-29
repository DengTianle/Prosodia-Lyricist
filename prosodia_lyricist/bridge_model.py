"""Contrastive melody trunk → autoregressive syllable stress/length templates."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .bridge_data import BOS, EOS, LINE_END, PAD, VOCAB
from .melody_encoder.encoding import MELODY_REPRESENTATION
from .melody_encoder.modeling import MelodyTransformerEncoder, SinusoidalPositionalEncoding

PROSODY_RULES = "ipa_binary_stress_diphthong_length_v2"


class ProsodyBridge(nn.Module):
    def __init__(
        self,
        melody_config,
        *,
        max_syllables=40,
        lines_per_window=1,
        max_notes=None,
        d_model=256,
        num_heads=4,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        provenance=None,
    ):
        super().__init__()
        for name, value in {
            "max_syllables": max_syllables,
            "lines_per_window": lines_per_window,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
        }.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads or not 0 <= dropout < 1:
            raise ValueError("Invalid bridge attention dimensions or dropout")
        self.melody_config = {**melody_config, "projection_dim": None}
        encoder_limit = self.melody_config.get("max_length", 4096)
        self.max_notes = encoder_limit if max_notes is None else max_notes
        if not isinstance(self.max_notes, int) or not 1 <= self.max_notes <= encoder_limit:
            raise ValueError("max_notes must be positive and fit the melody encoder")
        self.decoder_config = dict(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.max_syllables, self.lines_per_window = max_syllables, lines_per_window
        self.max_target_length = lines_per_window * (max_syllables + 1) + 1
        self.provenance = provenance or {}
        self.melody_encoder = MelodyTransformerEncoder(**self.melody_config)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.melody_encoder.d_model),
            nn.Linear(self.melody_encoder.d_model, d_model),
            nn.GELU(),
        )
        # Add phrase identities AFTER the pretrained tower, preserving its input distribution.
        self.line_embedding = nn.Embedding(lines_per_window + 1, d_model, padding_idx=0)
        self.token_embedding = nn.Embedding(len(VOCAB), d_model, padding_idx=PAD)
        self.positions = SinusoidalPositionalEncoding(d_model, self.max_target_length)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(d_model))
        # PyTorch clones the initial layer; initialize each decoder layer independently.
        for parameter in self.decoder.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        self.output = nn.Linear(d_model, len(VOCAB))
        self.melody_frozen = False

    def set_melody_trainable(self, trainable):
        self.melody_frozen = not trainable
        self.melody_encoder.requires_grad_(trainable)
        self.melody_encoder.train(self.training and trainable)

    def train(self, mode=True):
        super().train(mode)
        if self.melody_frozen:
            self.melody_encoder.eval()
        return self

    def encode(self, melody_features, melody_attention_mask, note_line_ids, line_counts):
        if melody_features.ndim != 3 or melody_features.shape[1] > self.max_notes:
            raise ValueError("Melody window exceeds bridge note limit or has an invalid shape")
        if melody_attention_mask.shape != melody_features.shape[:2] or (
            note_line_ids.shape != melody_attention_mask.shape
        ):
            raise ValueError("Melody masks and phrase IDs must align with notes")
        if (
            line_counts.shape != (len(melody_features),)
            or ((line_counts < 1) | (line_counts > self.lines_per_window)).any()
        ):
            raise ValueError("Invalid number of phrases in bridge window")
        active = melody_attention_mask.bool()
        if (
            not active.any(dim=1).all()
            or (note_line_ids[~active] != 0).any()
            or (((note_line_ids < 1) | (note_line_ids > line_counts[:, None])) & active).any()
        ):
            raise ValueError("Invalid note phrase membership or empty melody")
        with torch.no_grad() if self.melody_frozen else nullcontext():
            notes = self.melody_encoder.encode(
                melody_features, melody_attention_mask, project=False
            ).note_embeddings
        return self.adapter(notes) + self.line_embedding(note_line_ids)

    def decode(self, tokens, memory, melody_attention_mask):
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device).triu(1).bool()
        hidden = self.decoder(
            self.positions(self.token_embedding(tokens)),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tokens.eq(PAD),
            memory_key_padding_mask=~melody_attention_mask.bool(),
        )
        return self.output(hidden)

    def forward(self, melody_features, melody_attention_mask, note_line_ids, line_counts, labels):
        if (
            labels.ndim != 2
            or labels.shape[0] != len(melody_features)
            or (labels.shape[1] > self.max_target_length)
        ):
            raise ValueError("Invalid bridge target shape/length")
        if not labels.ne(-100).any():
            raise ValueError("No bridge target labels")
        memory = self.encode(melody_features, melody_attention_mask, note_line_ids, line_counts)
        tokens = torch.full_like(labels, PAD)
        tokens[:, 0] = BOS
        tokens[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), PAD)
        logits = self.decode(tokens, memory, melody_attention_mask)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)

    @torch.inference_mode()
    def generate(self, melody_features, melody_attention_mask, note_line_ids, line_counts):
        """Greedy templates with exactly the supplied number of nonempty phrases.

        Phrase-end predictions learn syllable counts independently of note counts.
        At the configured syllable cap, boundaries are forced and explicitly reported.
        """
        memory = self.encode(melody_features, melody_attention_mask, note_line_ids, line_counts)
        size, device = len(memory), memory.device
        tokens = torch.full((size, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(size, dtype=torch.bool, device=device)
        counts = torch.zeros(size, dtype=torch.long, device=device)
        phrases = torch.zeros_like(counts)
        forced = [[] for _ in range(size)]
        for _ in range(self.max_target_length):
            scores = self.decode(tokens, memory, melody_attention_mask)[:, -1].clone()
            scores[:, [PAD, BOS, EOS]] = -torch.inf
            scores[counts.eq(0), LINE_END] = -torch.inf
            next_token = scores.argmax(dim=-1)
            capped = counts.ge(self.max_syllables) & ~finished
            for row in (capped & next_token.ne(LINE_END)).nonzero().flatten().tolist():
                forced[row].append(int(phrases[row]))
            next_token[capped] = LINE_END
            next_token[phrases.eq(line_counts)] = EOS
            next_token[finished] = PAD
            ending = next_token.eq(LINE_END)
            counts = torch.where(ending, 0, counts + next_token.ge(4))
            phrases += ending
            finished |= next_token.eq(EOS)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            if finished.all():
                break
        if not finished.all():
            raise RuntimeError("Bridge generation did not complete within its structural limit")
        return SimpleNamespace(sequences=tokens, forced_line_endings=forced)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "melody_representation": MELODY_REPRESENTATION,
            "prosody_rules": PROSODY_RULES,
            "vocabulary": VOCAB,
            "melody_config": self.melody_config,
            "decoder_config": self.decoder_config,
            "max_syllables": self.max_syllables,
            "lines_per_window": self.lines_per_window,
            "max_notes": self.max_notes,
            "provenance": self.provenance,
        }
        (directory / "bridge.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save(self.state_dict(), directory / "bridge_weights.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "bridge.json").read_text(encoding="utf-8"))
        if (
            meta["format_version"] != 1
            or meta["vocabulary"] != VOCAB
            or (meta["melody_representation"] != MELODY_REPRESENTATION)
            or meta["prosody_rules"] != PROSODY_RULES
        ):
            raise ValueError("Incompatible bridge checkpoint format/feature rules")
        model = cls(
            meta["melody_config"],
            **meta["decoder_config"],
            max_syllables=meta["max_syllables"],
            lines_per_window=meta["lines_per_window"],
            max_notes=meta["max_notes"],
            provenance=meta["provenance"],
        )
        model.load_state_dict(
            torch.load(directory / "bridge_weights.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        return model
