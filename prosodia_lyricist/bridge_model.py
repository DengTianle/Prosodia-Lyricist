"""Contrastive melody trunk → autoregressive syllable stress/length templates."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .bridge_data import BOS, FIRST_LINE, PAD, PAIR_OFFSET, PAIRS, SLOT, bridge_vocabulary
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
        self.max_target_length = lines_per_window * (max_syllables + 1)
        self.vocabulary = bridge_vocabulary(lines_per_window)
        self.provenance = provenance or {}
        self.melody_encoder = MelodyTransformerEncoder(**self.melody_config)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.melody_encoder.d_model),
            nn.Linear(self.melody_encoder.d_model, d_model),
            nn.GELU(),
        )
        # Add phrase identities AFTER the pretrained tower, preserving its input distribution.
        self.line_embedding = nn.Embedding(lines_per_window + 1, d_model, padding_idx=0)
        self.token_embedding = nn.Embedding(len(self.vocabulary), d_model, padding_idx=PAD)
        self.slot_embedding = nn.Embedding(max_syllables + 1, d_model, padding_idx=0)
        self.count_embedding = nn.Embedding(max_syllables + 1, d_model, padding_idx=0)
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
        # Only prosody pairs can be predicted. Line markers come from the skeleton.
        self.output = nn.Linear(d_model, len(PAIRS))
        self.count_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, max_syllables))
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

    def predict_counts(self, memory, note_line_ids, line_counts):
        line_ids = torch.arange(1, int(line_counts.max()) + 1, device=memory.device)
        membership = note_line_ids[:, None].eq(line_ids[None, :, None])
        active = line_ids[None].le(line_counts[:, None])
        if (membership.sum(-1).eq(0) & active).any():
            raise ValueError("Every skeleton line must have melody notes")
        weights = membership.to(memory.dtype)
        pooled = weights.bmm(memory) / weights.sum(-1, keepdim=True).clamp_min(1)
        return self.count_head(pooled)

    def resolve_counts(self, count_logits, line_counts, supplied=None, *, training=False):
        active = torch.arange(count_logits.shape[1], device=line_counts.device)[None].lt(
            line_counts[:, None]
        )
        predicted = (count_logits.argmax(-1) + 1).masked_fill(~active, 0)
        if supplied is None:
            if training:
                raise ValueError("Training requires a DALI syllable-count skeleton")
            return predicted, predicted
        if supplied.shape != predicted.shape or supplied.dtype not in (torch.int32, torch.int64):
            raise ValueError("Skeleton syllable_counts must be integer counts for each line")
        if (
            ((supplied < 0) | (supplied > self.max_syllables)).any()
            or (supplied[~active].ne(0).any())
            or (training and supplied[active].eq(0).any())
        ):
            raise ValueError("Invalid skeleton syllable counts or nonzero padding")
        return torch.where(supplied.ne(0), supplied, predicted), predicted

    def make_skeleton(self, counts, *, width=None):
        """Fixed line prefixes and known slots; local IDs restart at each melody window."""
        rows = []
        for row in counts.tolist():
            tokens, lines, slots, sizes = [], [], [], []
            for index, count in enumerate(row):
                if not count:
                    continue
                tokens.extend([FIRST_LINE + index] + [SLOT] * count)
                lines.extend([index + 1] * (count + 1))
                slots.extend(range(count + 1))
                sizes.extend([count] * (count + 1))
            rows.append((tokens, lines, slots, sizes))
        required = max(len(row[0]) for row in rows)
        width = required if width is None else width
        if not required <= width <= self.max_target_length:
            raise ValueError("Skeleton exceeds target length or does not fit the labels")
        return {
            key: torch.tensor(
                [row[index] + [0] * (width - len(row[index])) for row in rows],
                dtype=torch.long,
                device=counts.device,
            )
            for index, key in enumerate(("tokens", "line_ids", "slot_ids", "counts"))
        }

    def decode(self, tokens, memory, melody_attention_mask, skeleton):
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device).triu(1).bool()
        queries = (
            self.token_embedding(tokens)
            + self.line_embedding(skeleton["line_ids"])
            + self.slot_embedding(skeleton["slot_ids"])
            + self.count_embedding(skeleton["counts"])
        )
        hidden = self.decoder(
            self.positions(queries),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=skeleton["tokens"].eq(PAD),
            memory_key_padding_mask=~melody_attention_mask.bool(),
        )
        return self.output(hidden)

    def forward(
        self,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        labels,
        syllable_counts,
    ):
        if (
            labels.ndim != 2
            or labels.shape[0] != len(melody_features)
            or (labels.shape[1] > self.max_target_length)
        ):
            raise ValueError("Invalid bridge target shape/length")
        if not labels.ne(-100).any():
            raise ValueError("No bridge target labels")
        memory = self.encode(melody_features, melody_attention_mask, note_line_ids, line_counts)
        count_logits = self.predict_counts(memory, note_line_ids, line_counts)
        counts, _ = self.resolve_counts(count_logits, line_counts, syllable_counts, training=True)
        skeleton = self.make_skeleton(counts, width=labels.shape[1])
        slots = skeleton["tokens"].eq(SLOT)
        structure = skeleton["tokens"].masked_fill(skeleton["tokens"].eq(PAD), -100)
        if (
            labels[~slots].ne(structure[~slots]).any()
            or ((labels[slots] < PAIR_OFFSET) | (labels[slots] >= FIRST_LINE)).any()
        ):
            raise ValueError("Prosody labels must fit the supplied line skeleton")
        tokens = torch.full_like(labels, PAD)
        tokens[:, 0] = BOS
        tokens[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), PAD)
        logits = self.decode(tokens, memory, melody_attention_mask, skeleton)
        pair_targets = (labels - PAIR_OFFSET).masked_fill(~slots, -100)
        count_targets = (counts - 1).masked_fill(counts.eq(0), -100)
        components = {
            "prosody": F.cross_entropy(logits.flatten(0, 1), pair_targets.flatten()),
            "counts": F.cross_entropy(count_logits.flatten(0, 1), count_targets.flatten()),
        }
        return SimpleNamespace(
            loss=sum(components.values()),
            logits=logits,
            count_logits=count_logits,
            loss_components=components,
        )

    @torch.inference_mode()
    def generate(
        self,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        syllable_counts=None,
    ):
        """Fill a fixed skeleton. A zero/omitted count is predicted once per melody line."""
        memory = self.encode(melody_features, melody_attention_mask, note_line_ids, line_counts)
        count_logits = self.predict_counts(memory, note_line_ids, line_counts)
        counts, predicted_counts = self.resolve_counts(count_logits, line_counts, syllable_counts)
        skeleton = self.make_skeleton(counts)
        size, device = len(memory), memory.device
        tokens = torch.full((size, 1), BOS, dtype=torch.long, device=device)
        for step in range(skeleton["tokens"].shape[1]):
            next_token = skeleton["tokens"][:, step].clone()
            slots = next_token.eq(SLOT)
            if slots.any():
                prefix = {key: value[:, : step + 1] for key, value in skeleton.items()}
                scores = self.decode(tokens, memory, melody_attention_mask, prefix)[:, -1]
                next_token[slots] = scores[slots].argmax(-1) + PAIR_OFFSET
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        return SimpleNamespace(
            sequences=tokens[:, 1:],
            syllable_counts=counts,
            predicted_syllable_counts=predicted_counts,
        )

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 2,
            "target_scheme": "line_skeleton_v2",
            "melody_representation": MELODY_REPRESENTATION,
            "prosody_rules": PROSODY_RULES,
            "vocabulary": self.vocabulary,
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
        if meta["format_version"] == 1:
            raise ValueError("The LINE_END bridge uses format v1; retrain for the v2 line skeleton")
        if (
            meta["format_version"] != 2
            or meta.get("target_scheme") != "line_skeleton_v2"
            or meta["vocabulary"] != bridge_vocabulary(meta["lines_per_window"])
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
