"""Contrastive melody trunk → autoregressive syllable stress/length templates."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .bridge_data import (
    BOS,
    FIRST_LINE,
    PAD,
    PAIR_OFFSET,
    PAIRS,
    SLOT,
    SONG_FEATURE_DIM,
    SONG_FEATURE_SCHEME,
    bridge_vocabulary,
)
from .bridge_decoding import DecoderCache
from .melody_encoder.encoding import MELODY_REPRESENTATION
from .melody_encoder.modeling import MelodyTransformerEncoder, SinusoidalPositionalEncoding

PROSODY_RULES = "ipa_binary_stress_diphthong_length_v2"
TARGET_SCHEME = "line_skeleton_v5"
OUTPUT_LABELS = {"strength": ["strong", "weak"], "length": ["long", "short"]}


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
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        provenance=None,
        output_design="separate",
        bridge_scope="window",
        encoder_lines_per_window=None,
        max_window_notes=None,
        max_song_notes=2048,
        max_song_lines=256,
        max_target_length=None,
        song_encoder_layers=2,
    ):
        super().__init__()
        if encoder_lines_per_window is not None:
            lines_per_window = encoder_lines_per_window
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
        if output_design not in ("separate", "joint_pair"):
            raise ValueError("Unknown bridge output design")
        if bridge_scope not in ("window", "song") or (
            bridge_scope == "song" and output_design != "separate"
        ):
            raise ValueError("Song bridges require separate heads and a supported bridge scope")
        self.bridge_scope = bridge_scope
        self.output_design = output_design
        self.target_scheme = (
            TARGET_SCHEME if bridge_scope == "song" else
            "line_skeleton_v4" if output_design == "separate" else "line_skeleton_v3"
        )
        self.melody_config = {**melody_config, "projection_dim": None}
        encoder_limit = self.melody_config.get("max_length", 4096)
        if max_window_notes is not None:
            if max_notes is not None and max_notes != max_window_notes:
                raise ValueError("Conflicting encoder window note limits")
            max_notes = max_window_notes
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
        self.encoder_lines_per_window = lines_per_window
        self.max_song_notes, self.max_song_lines = max_song_notes, max_song_lines
        for name, value in (("max_song_notes", max_song_notes), ("max_song_lines", max_song_lines)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(song_encoder_layers, int) or song_encoder_layers < 0:
            raise ValueError("song_encoder_layers must be a nonnegative integer")
        self.song_encoder_layers = song_encoder_layers if bridge_scope == "song" else 0
        self.max_lines = max_song_lines if bridge_scope == "song" else lines_per_window
        self.max_target_length = (
            self.max_lines * (max_syllables + 1) if max_target_length is None else max_target_length
        )
        if not isinstance(self.max_target_length, int) or self.max_target_length < 2:
            raise ValueError("max_target_length must be an integer of at least two")
        self.vocabulary = bridge_vocabulary(self.max_lines)
        self.provenance = provenance or {}
        self.melody_encoder = MelodyTransformerEncoder(**self.melody_config)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.melody_encoder.d_model),
            nn.Linear(self.melody_encoder.d_model, d_model),
            nn.GELU(),
        )
        # Add phrase identities AFTER the pretrained tower, preserving its input distribution.
        self.line_embedding = nn.Embedding(self.max_lines + 1, d_model, padding_idx=0)
        if bridge_scope == "song":
            self.source_positions = SinusoidalPositionalEncoding(d_model, max_song_notes)
            self.song_features = nn.Sequential(
                nn.Linear(SONG_FEATURE_DIM, d_model), nn.GELU(),
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            )
            self.source_norm = nn.LayerNorm(d_model)
            if song_encoder_layers:
                song_layer = nn.TransformerEncoderLayer(
                    d_model, num_heads, dim_feedforward, dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                self.song_encoder = nn.TransformerEncoder(
                    song_layer, song_encoder_layers, norm=nn.LayerNorm(d_model),
                    enable_nested_tensor=False,
                )
                for parameter in self.song_encoder.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
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
        # Paired tokens remain the feedback/serialization format, not a joint classifier.
        if output_design == "separate":
            self.strength_head = nn.Linear(d_model, 2)
            self.length_head = nn.Linear(d_model, 2)
        else:
            # Preserve old checkpoints exactly; two linear heads cannot reproduce an
            # arbitrary joint four-class distribution.
            self.output = nn.Linear(d_model, len(PAIRS))
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

    @property
    def dataset_options(self):
        return dict(
            bridge_scope=self.bridge_scope, lines_per_window=self.encoder_lines_per_window,
            max_notes=self.max_notes, max_song_notes=self.max_song_notes,
            max_song_lines=self.max_song_lines, max_target_length=self.max_target_length,
        )

    def encode(
        self, melody_features, melody_attention_mask, note_line_ids, line_counts,
        song_features=None,
    ):
        note_limit = self.max_song_notes if self.bridge_scope == "song" else self.max_notes
        if melody_features.ndim != 3 or melody_features.shape[1] > note_limit:
            raise ValueError("Melody exceeds bridge note limit or has an invalid shape")
        if melody_attention_mask.shape != melody_features.shape[:2] or (
            note_line_ids.shape != melody_attention_mask.shape
        ):
            raise ValueError("Melody masks and phrase IDs must align with notes")
        if (
            line_counts.shape != (len(melody_features),)
            or ((line_counts < 1) | (line_counts > self.max_lines)).any()
        ):
            raise ValueError("Invalid number of phrases in bridge window")
        active = melody_attention_mask.bool()
        if (
            not active.any(dim=1).all()
            or (note_line_ids[~active] != 0).any()
            or (((note_line_ids < 1) | (note_line_ids > line_counts[:, None])) & active).any()
        ):
            raise ValueError("Invalid note phrase membership or empty melody")
        counts = self.note_counts(note_line_ids, line_counts)
        active_lines = torch.arange(counts.shape[1], device=counts.device)[None].lt(
            line_counts[:, None]
        )
        if (counts.eq(0) & active_lines).any():
            raise ValueError("Every skeleton line must have melody notes")
        if self.bridge_scope == "song":
            if song_features is None or song_features.shape != (
                *melody_features.shape[:2], SONG_FEATURE_DIM
            ) or not torch.isfinite(song_features[active]).all():
                raise ValueError("Song bridge requires aligned, finite global melody features")
            # Positions and window packing require consecutive notes and ordered phrase IDs.
            if not torch.equal(
                active, torch.arange(active.shape[1], device=active.device)[None]
                < active.sum(-1, keepdim=True)
            ) or ((note_line_ids[:, 1:] < note_line_ids[:, :-1]) & active[:, 1:]).any():
                raise ValueError("Song notes must be ordered with trailing padding")
            notes = self.encode_windows(melody_features, active, note_line_ids, line_counts)
            auxiliary = self.song_features(song_features.masked_fill(~active[..., None], 0))
            memory = self.source_norm(
                self.source_positions(self.adapter(notes))
                + self.line_embedding(note_line_ids) + auxiliary
            )
            if self.song_encoder_layers:
                memory = self.song_encoder(memory, src_key_padding_mask=~active)
            return memory.masked_fill(~active[..., None], 0)
        if song_features is not None:
            raise ValueError("Window bridge does not accept song features")
        with torch.no_grad() if self.melody_frozen else nullcontext():
            notes = self.melody_encoder.encode(
                melody_features, melody_attention_mask, project=False
            ).note_embeddings
        return self.adapter(notes) + self.line_embedding(note_line_ids)

    def encode_windows(self, features, active, line_ids, line_counts):
        """Batch all real local windows, then scatter note vectors back into songs."""
        windows_per_song = (line_counts + self.encoder_lines_per_window - 1) // (
            self.encoder_lines_per_window
        )
        offsets = windows_per_song.cumsum(0) - windows_per_song
        song_rows = active.nonzero(as_tuple=True)[0]
        window_rows = offsets[song_rows] + (line_ids[active] - 1) // self.encoder_lines_per_window
        lengths = torch.bincount(window_rows, minlength=int(windows_per_song.sum()))
        width = int(lengths.max())
        if width > self.max_notes:
            raise ValueError("Encoder window exceeds bridge note limit; no truncation")
        starts = lengths.cumsum(0) - lengths
        local_positions = torch.arange(len(window_rows), device=features.device) - (
            starts.repeat_interleave(lengths)
        )
        packed = features.new_zeros(len(lengths), width, features.shape[-1])
        packed[window_rows, local_positions] = features[active]
        mask = torch.arange(width, device=features.device)[None] < lengths[:, None]
        with torch.no_grad() if self.melody_frozen else nullcontext():
            encoded = self.melody_encoder.encode(packed, mask, project=False).note_embeddings
        notes = encoded.new_zeros(*features.shape[:2], encoded.shape[-1])
        notes[active] = encoded[window_rows, local_positions]
        return notes

    @staticmethod
    def note_counts(note_line_ids, line_counts):
        """Count actual notes per phrase; padding has phrase ID zero."""
        line_ids = torch.arange(1, int(line_counts.max()) + 1, device=note_line_ids.device)
        return note_line_ids[:, None].eq(line_ids[None, :, None]).sum(-1)

    def validate_counts(self, counts, line_counts):
        if counts.dtype not in (torch.int32, torch.int64) or counts.shape != (
            len(line_counts), int(line_counts.max())
        ):
            raise ValueError("Skeleton syllable_counts must be integer counts for each line")
        active = torch.arange(counts.shape[1], device=line_counts.device)[None].lt(
            line_counts[:, None]
        )
        if (
            ((counts < 0) | (counts > self.max_syllables)).any()
            or counts[~active].ne(0).any()
            or counts[active].eq(0).any()
        ):
            raise ValueError("Invalid skeleton syllable counts or nonzero padding")

    def make_skeleton(self, counts, *, width=None):
        """Fixed line prefixes and known slots within the model's window/song scope."""
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

    def query_embeddings(self, tokens, skeleton, offset=0):
        queries = (
            self.token_embedding(tokens)
            + self.line_embedding(skeleton["line_ids"])
            + self.slot_embedding(skeleton["slot_ids"])
            + self.count_embedding(skeleton["counts"])
        )
        return queries + self.positions.encoding[:, offset : offset + tokens.shape[1]].to(
            dtype=queries.dtype
        )

    def decode(self, tokens, memory, melody_attention_mask, skeleton):
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device).triu(1).bool()
        hidden = self.decoder(
            self.query_embeddings(tokens, skeleton),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=skeleton["tokens"].eq(PAD),
            memory_key_padding_mask=~melody_attention_mask.bool(),
        )
        return self.classify(hidden)

    def classify(self, hidden):
        if self.output_design == "joint_pair":
            logits = self.output(hidden)
            pairs = logits.unflatten(-1, (2, 2))
            strength_logits = pairs.logsumexp(-1)
            length_logits = pairs.logsumexp(-2)
        else:
            strength_logits = self.strength_head(hidden)
            length_logits = self.length_head(hidden)
            # PAIRS order: strong-long, strong-short, weak-long, weak-short.
            # Softmax of these scores is the product of the two head distributions.
            logits = (strength_logits.unsqueeze(-1) + length_logits.unsqueeze(-2)).flatten(-2)
        return SimpleNamespace(
            logits=logits, strength_logits=strength_logits, length_logits=length_logits
        )

    def forward(
        self,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        labels,
        syllable_counts,
        song_features=None,
    ):
        if (
            labels.ndim != 2
            or labels.shape[0] != len(melody_features)
            or (labels.shape[1] > self.max_target_length)
        ):
            raise ValueError("Invalid bridge target shape/length")
        if not labels.ne(-100).any():
            raise ValueError("No bridge target labels")
        memory = self.encode(
            melody_features, melody_attention_mask, note_line_ids, line_counts, song_features
        )
        self.validate_counts(syllable_counts, line_counts)
        skeleton = self.make_skeleton(syllable_counts, width=labels.shape[1])
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
        output = self.decode(tokens, memory, melody_attention_mask, skeleton)
        pair_targets = (labels - PAIR_OFFSET).masked_fill(~slots, -100)
        if self.output_design == "separate":
            strength_targets = (pair_targets // 2).masked_fill(~slots, -100)
            length_targets = (pair_targets % 2).masked_fill(~slots, -100)
            components = {
                "strength": F.cross_entropy(
                    output.strength_logits.flatten(0, 1), strength_targets.flatten()
                ),
                "length": F.cross_entropy(
                    output.length_logits.flatten(0, 1), length_targets.flatten()
                ),
            }
        else:
            components = {
                "prosody": F.cross_entropy(output.logits.flatten(0, 1), pair_targets.flatten())
            }
        return SimpleNamespace(
            loss=sum(components.values()),
            logits=output.logits,
            strength_logits=output.strength_logits,
            length_logits=output.length_logits,
            loss_components=components,
        )

    @torch.inference_mode()
    def generate(
        self,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        song_features=None,
        *,
        use_cache=None,
    ):
        """Fill exactly one prosody slot per note in each supplied melody phrase."""
        memory = self.encode(
            melody_features, melody_attention_mask, note_line_ids, line_counts, song_features
        )
        counts = self.note_counts(note_line_ids, line_counts)
        if counts.gt(self.max_syllables).any():
            raise ValueError(
                f"Phrase note count exceeds bridge slot limit {self.max_syllables}; no truncation"
            )
        self.validate_counts(counts, line_counts)
        skeleton = self.make_skeleton(counts)
        size, device = len(memory), memory.device
        tokens = torch.full((size, 1), BOS, dtype=torch.long, device=device)
        use_cache = self.bridge_scope == "song" if use_cache is None else use_cache
        cache = (
            DecoderCache(self.decoder, memory, melody_attention_mask, skeleton["tokens"].shape[1])
            if use_cache else None
        )
        for step in range(skeleton["tokens"].shape[1]):
            next_token = skeleton["tokens"][:, step].clone()
            slots = next_token.eq(SLOT)
            if cache is not None:
                current = {key: value[:, step : step + 1] for key, value in skeleton.items()}
                # Forced line prefixes also advance the cache; they condition later slots.
                hidden = cache.step(
                    self.query_embeddings(tokens[:, -1:], current, offset=step),
                    next_token.ne(PAD),
                )
                output = self.classify(hidden)
            if slots.any():
                if cache is None:
                    prefix = {key: value[:, : step + 1] for key, value in skeleton.items()}
                    output = self.decode(tokens, memory, melody_attention_mask, prefix)
                if self.output_design == "separate":
                    strength = output.strength_logits[slots, -1].argmax(-1)
                    length = output.length_logits[slots, -1].argmax(-1)
                    next_token[slots] = PAIR_OFFSET + 2 * strength + length
                else:
                    next_token[slots] = output.logits[slots, -1].argmax(-1) + PAIR_OFFSET
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        return SimpleNamespace(
            sequences=tokens[:, 1:],
            syllable_counts=counts,
        )

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": (
                5 if self.bridge_scope == "song" else 4 if self.output_design == "separate" else 3
            ),
            "target_scheme": self.target_scheme,
            "output_design": self.output_design,
            "output_labels": OUTPUT_LABELS,
            "training_count_source": "ipa",
            "inference_count_source": "notes",
            "melody_representation": MELODY_REPRESENTATION,
            "prosody_rules": PROSODY_RULES,
            "vocabulary": self.vocabulary,
            "melody_config": self.melody_config,
            "decoder_config": self.decoder_config,
            "max_syllables": self.max_syllables,
            "lines_per_window": self.lines_per_window,
            "max_notes": self.max_notes,
            "provenance": self.provenance,
            "bridge_scope": self.bridge_scope,
            "max_target_length": self.max_target_length,
        }
        if self.bridge_scope == "song":
            metadata.update(
                encoder_lines_per_window=self.encoder_lines_per_window,
                max_window_notes=self.max_notes,
                max_song_notes=self.max_song_notes, max_song_lines=self.max_song_lines,
                song_encoder_layers=self.song_encoder_layers,
                song_feature_scheme=SONG_FEATURE_SCHEME, song_feature_dim=SONG_FEATURE_DIM,
                source_position_scheme="global_note_sinusoidal_line_embedding_v1",
            )
        (directory / "bridge.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save(self.state_dict(), directory / "bridge_weights.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "bridge.json").read_text(encoding="utf-8"))
        if meta["format_version"] == 1:
            raise ValueError("The LINE_END bridge uses format v1; retrain for the current skeleton")
        version = meta["format_version"]
        if version == 5 and not {
            "max_song_notes", "max_song_lines", "max_target_length", "song_encoder_layers",
            "song_feature_scheme", "song_feature_dim", "source_position_scheme",
            "encoder_lines_per_window", "max_window_notes",
        }.issubset(meta):
            raise ValueError("Incomplete song bridge checkpoint architecture/feature rules")
        if (
            version not in (2, 3, 4, 5)
            or meta.get("target_scheme") != f"line_skeleton_v{version}"
            or (version >= 3 and meta.get("inference_count_source") != "notes")
            or (version >= 4 and (
                meta.get("output_design") != "separate"
                or meta.get("output_labels") != OUTPUT_LABELS
                or meta.get("training_count_source") != "ipa"
            ))
            or meta["vocabulary"] != bridge_vocabulary(
                meta["max_song_lines"] if version == 5 else meta["lines_per_window"]
            )
            or (meta["melody_representation"] != MELODY_REPRESENTATION)
            or meta["prosody_rules"] != PROSODY_RULES
            or (version < 5 and meta.get("bridge_scope", "window") != "window")
            or (version == 5 and (
                meta.get("bridge_scope") != "song"
                or meta.get("song_feature_scheme") != SONG_FEATURE_SCHEME
                or meta.get("song_feature_dim") != SONG_FEATURE_DIM
                or meta.get("source_position_scheme") != "global_note_sinusoidal_line_embedding_v1"
                or meta.get("encoder_lines_per_window") != meta["lines_per_window"]
                or meta.get("max_window_notes") != meta["max_notes"]
            ))
        ):
            raise ValueError("Incompatible bridge checkpoint format/feature rules")
        model = cls(
            meta["melody_config"],
            **meta["decoder_config"],
            max_syllables=meta["max_syllables"],
            lines_per_window=meta["lines_per_window"],
            max_notes=meta["max_notes"],
            provenance=meta["provenance"],
            output_design="separate" if version >= 4 else "joint_pair",
            bridge_scope="song" if version == 5 else "window",
            max_target_length=meta.get("max_target_length"),
            **({key: meta[key] for key in (
                "max_song_notes", "max_song_lines", "song_encoder_layers",
            )} if version == 5 else {}),
        )
        state = torch.load(directory / "bridge_weights.pt", map_location="cpu", weights_only=True)
        if meta["format_version"] == 2:
            # The v2 prosody decoder is identical; discard only the removed count-head weights.
            for key in (
                "count_head.0.weight", "count_head.0.bias",
                "count_head.1.weight", "count_head.1.bias",
            ):
                if key not in state:
                    raise ValueError(f"Incomplete v2 bridge checkpoint: missing {key}")
                state.pop(key)
        model.load_state_dict(state, strict=True)
        return model
