"""One global melody encoder with optional meter and contrastive conditioning."""

import json
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .bridge_alignment import ALIGNMENT_FORMAT, ALIGNMENT_POLICIES, ALIGNMENT_SCHEME
from .bridge_data import PAD, PAIR_OFFSET, SLOT, bridge_vocabulary, prosody_pairs
from .ipa import PAPER_RULES, REPO_RULES, REPO_STRESSES
from .melody_encoder.encoding import MELODY_REPRESENTATION
from .melody_encoder.modeling import MelodyTransformerEncoder
from .melody_encoder.octuple import REPRESENTATION, codec_config, codec_metadata
from .melody_encoder.octuple_model import OctupleMelodyEncoder
from .meter import METER_FEATURE_DIM, METER_FEATURE_SCHEME

FORMAT_VERSION = 10
SOURCE_SCHEME = "global_note_177d_v1"
TARGET_SCHEME = "aligned_note_heads_v1"


class ProsodyBridge(nn.Module):
    # These describe the sole supported design, including evaluation report semantics.
    bridge_scope = "song"
    query_mode = "note"
    query_attention = "none"
    output_design = "separate"
    target_scheme = TARGET_SCHEME

    def __init__(
        self,
        melody_config,
        *,
        max_syllables=40,
        max_song_notes=2048,
        max_song_lines=256,
        max_target_length=4096,
        strength_labels=REPO_STRESSES,
        alignment_policy="strict",
        meter_conditioning=False,
        contrastive_config=None,
        provenance=None,
        representation=MELODY_REPRESENTATION,
        octuple_config=None,
        predict_note_roles=False,
        role_loss_weight=0.5,
        role_threshold=0.5,
        role_class_weights=None,
        canonical_timing=False,
    ):
        super().__init__()
        for name, value in dict(
            max_syllables=max_syllables,
            max_song_notes=max_song_notes,
            max_song_lines=max_song_lines,
            max_target_length=max_target_length,
        ).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_target_length < 2:
            raise ValueError("max_target_length must be at least two")
        if alignment_policy not in ALIGNMENT_POLICIES:
            raise ValueError("alignment_policy must be strict or prefix")
        if not isinstance(meter_conditioning, bool):
            raise ValueError("meter_conditioning must be boolean")
        if representation not in (MELODY_REPRESENTATION, REPRESENTATION):
            raise ValueError("Unsupported melody representation")
        if type(predict_note_roles) is not bool:
            raise ValueError("predict_note_roles must be boolean")
        if (
            not isinstance(role_loss_weight, (int, float))
            or isinstance(role_loss_weight, bool)
            or not math.isfinite(role_loss_weight)
            or role_loss_weight <= 0
            or not isinstance(role_threshold, (int, float))
            or isinstance(role_threshold, bool)
            or not 0 < role_threshold < 1
        ):
            raise ValueError("Positive role_loss_weight and role_threshold in (0, 1) required")
        self.representation = representation
        if type(canonical_timing) is not bool:
            raise ValueError("canonical_timing must be boolean")
        self.canonical_timing = canonical_timing or representation == REPRESENTATION
        self.source_scheme = (
            REPRESENTATION
            if representation == REPRESENTATION
            else "global_note_177d_canonical_meter_v1"
            if self.canonical_timing
            else SOURCE_SCHEME
        )
        self.octuple_config = (
            codec_config(octuple_config) if representation == REPRESENTATION else None
        )
        if representation != REPRESENTATION and octuple_config is not None:
            raise ValueError("Octuple options require the octuple representation")
        self.predict_note_roles = predict_note_roles
        self.role_loss_weight, self.role_threshold = role_loss_weight, role_threshold
        self.melody_config = {**melody_config, "projection_dim": None}
        if max_song_notes > self.melody_config.get("max_length", 4096):
            raise ValueError("max_song_notes must fit the global melody encoder max_length")
        self.max_syllables, self.max_song_notes = max_syllables, max_song_notes
        self.max_song_lines, self.max_target_length = max_song_lines, max_target_length
        self.max_lines = max_song_lines
        self.alignment_policy = alignment_policy
        self.strength_labels = tuple(strength_labels)
        self.pairs = prosody_pairs(self.strength_labels)
        self.first_line = PAIR_OFFSET + len(self.pairs)
        self.prosody_rules = REPO_RULES if self.strength_labels == REPO_STRESSES else PAPER_RULES
        self.vocabulary = bridge_vocabulary(self.max_lines, pairs=self.pairs)
        self.provenance = provenance or {}
        self.supervised_exposure = None
        self.melody_encoder = (
            OctupleMelodyEncoder(**self.melody_config, octuple_config=self.octuple_config)
            if representation == REPRESENTATION
            else MelodyTransformerEncoder(**self.melody_config)
        )
        width = self.melody_encoder.d_model
        self.line_embedding = nn.Embedding(self.max_lines + 1, width, padding_idx=0)
        self.head_dropout = nn.Dropout(self.melody_config.get("dropout", 0.1))
        self.strength_head = nn.Linear(width, len(self.strength_labels))
        self.length_head = nn.Linear(width, 2)
        self.role_head = nn.Linear(width, 2) if predict_note_roles else None
        if predict_note_roles:
            weights = [1.0, 1.0] if role_class_weights is None else role_class_weights
            if len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights):
                raise ValueError("role_class_weights must be positive [START, CONTINUE] weights")
            self.register_buffer("role_class_weights", torch.tensor(weights, dtype=torch.float32))
        self.meter_adapter = None
        if meter_conditioning:
            self.enable_meter()
        self.contrastive_encoder = None
        self.contrastive_config = None
        self.contrastive_frozen = True
        if contrastive_config is not None:
            self.add_contrastive(contrastive_config)
        self.melody_frozen = False

    @property
    def meter_conditioning(self):
        return self.meter_adapter is not None

    def enable_meter(self):
        """Add meter to a DALI checkpoint without changing its initial predictions."""
        if self.representation == REPRESENTATION:
            raise ValueError("Octuple includes meter; disable the legacy meter adapter")
        if self.meter_adapter is None:
            width = self.melody_encoder.d_model
            self.meter_adapter = nn.Sequential(
                nn.Linear(METER_FEATURE_DIM, width), nn.GELU(), nn.Linear(width, width)
            ).to(self.line_embedding.weight)
            nn.init.zeros_(self.meter_adapter[-1].weight)
            nn.init.zeros_(self.meter_adapter[-1].bias)

    def add_contrastive(self, config, state=None):
        """Fuse a whole-song contrastive tower; its audio/projection heads are unused."""
        if self.representation == REPRESENTATION:
            raise ValueError("Octuple does not support contrastive fusion")
        if self.contrastive_encoder is not None:
            raise ValueError("A contrastive encoder is already attached")
        config = {**config, "projection_dim": None}
        if self.max_song_notes > config.get("max_length", 4096):
            raise ValueError("max_song_notes must fit the contrastive encoder max_length")
        self.contrastive_config = config
        self.contrastive_encoder = MelodyTransformerEncoder(**config)
        if state is not None:
            self.contrastive_encoder.load_state_dict(state, strict=True)
        self.contrastive_adapter = nn.Sequential(
            nn.LayerNorm(2 * self.contrastive_encoder.d_model),
            nn.Linear(2 * self.contrastive_encoder.d_model, self.melody_encoder.d_model),
            nn.GELU(),
        )
        # A newly added branch starts as an exact residual identity.
        self.contrastive_gate = nn.Parameter(self.line_embedding.weight.new_zeros(()))
        self.contrastive_encoder.to(self.line_embedding.weight)
        self.contrastive_adapter.to(self.line_embedding.weight)
        self.set_contrastive_trainable(False)

    def set_melody_trainable(self, trainable):
        self.melody_frozen = not trainable
        self.melody_encoder.requires_grad_(trainable)
        self.melody_encoder.train(self.training and trainable)

    def set_contrastive_trainable(self, trainable):
        self.contrastive_frozen = not trainable
        if self.contrastive_encoder is not None:
            self.contrastive_encoder.requires_grad_(trainable)
            self.contrastive_encoder.train(self.training and trainable)

    def train(self, mode=True):
        super().train(mode)
        if self.melody_frozen:
            self.melody_encoder.eval()
        if self.contrastive_encoder is not None and self.contrastive_frozen:
            self.contrastive_encoder.eval()
        return self

    @property
    def dataset_options(self):
        return dict(
            max_song_notes=self.max_song_notes,
            max_song_lines=self.max_song_lines,
            max_target_length=self.max_target_length,
            strength_labels=self.strength_labels,
            alignment_policy=self.alignment_policy,
            representation=self.representation,
            octuple_config=self.octuple_config,
            predict_note_roles=self.predict_note_roles,
            canonical_timing=self.canonical_timing,
        )

    def encode(
        self,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        meter_features=None,
        meter_mask=None,
        timing_residuals=None,
    ):
        if melody_features.ndim != 3 or not 1 <= melody_features.shape[1] <= self.max_song_notes:
            raise ValueError("Melody exceeds bridge note limit or has an invalid shape")
        if melody_attention_mask.shape != melody_features.shape[:2] or (
            note_line_ids.shape != melody_attention_mask.shape
        ):
            raise ValueError("Melody masks and phrase IDs must align with notes")
        if (
            line_counts.shape != (len(melody_features),)
            or line_counts.dtype != torch.long
            or note_line_ids.dtype != torch.long
            or ((line_counts < 1) | (line_counts > self.max_lines)).any()
        ):
            raise ValueError("Invalid number of phrases or phrase ID type")
        active = melody_attention_mask.bool()
        if (
            not active.any(dim=1).all()
            or (note_line_ids[~active] != 0).any()
            or (((note_line_ids < 1) | (note_line_ids > line_counts[:, None])) & active).any()
        ):
            raise ValueError("Invalid note phrase membership or empty melody")
        if (
            not torch.equal(
                active,
                torch.arange(active.shape[1], device=active.device)[None]
                < active.sum(-1, keepdim=True),
            )
            or ((note_line_ids[:, 1:] < note_line_ids[:, :-1]) & active[:, 1:]).any()
        ):
            raise ValueError("Song notes must be ordered with trailing padding")
        counts = self.note_counts(note_line_ids, line_counts)
        active_lines = (
            torch.arange(counts.shape[1], device=counts.device)[None] < line_counts[:, None]
        )
        if (counts.eq(0) & active_lines).any():
            raise ValueError("Every skeleton line must have melody notes")
        if not torch.isfinite(melody_features[active]).all():
            raise ValueError("Melody features must be finite")
        features = melody_features.masked_fill(~active[..., None], 0)
        context = self.line_embedding(note_line_ids)
        if (meter_features is None) != (meter_mask is None):
            raise ValueError("Supply both meter_features and meter_mask, or neither")
        if meter_features is not None:
            if meter_features.shape != (*active.shape, METER_FEATURE_DIM) or (
                meter_mask.shape != active.shape
            ):
                raise ValueError("Meter features/mask must align with melody notes")
            observed = meter_mask.bool() & active
            if not torch.isfinite(meter_features[observed]).all():
                raise ValueError("Observed meter features must be finite")
            if self.meter_adapter is not None:
                meter = meter_features.masked_fill(~observed[..., None], 0)
                # Mask AFTER the adapter, so missing meter has exactly zero contribution.
                context = context + self.meter_adapter(meter).masked_fill(~observed[..., None], 0)
        if self.contrastive_encoder is not None:
            with torch.no_grad() if self.contrastive_frozen else nullcontext():
                encoded = self.contrastive_encoder.encode(features, active, project=False)
            pooled = encoded.pooled_embedding[:, None].expand_as(encoded.note_embeddings)
            fused = self.contrastive_adapter(torch.cat((encoded.note_embeddings, pooled), dim=-1))
            context = context + self.contrastive_gate.tanh() * fused
        # Even a frozen trunk must propagate gradients into trainable conditioning.
        extra = {}
        if self.representation == REPRESENTATION:
            if timing_residuals is None or timing_residuals.shape != (*active.shape, 3):
                raise ValueError("Octuple input requires timing_residuals")
            extra["timing_residuals"] = timing_residuals.masked_fill(~active[..., None], 0)
        elif timing_residuals is not None:
            raise ValueError("Legacy melody encoding does not accept timing residuals")
        notes = self.melody_encoder.encode(
            features,
            active,
            project=False,
            note_context=context,
            **extra,
        ).note_embeddings
        return notes.masked_fill(~active[..., None], 0)

    @staticmethod
    def note_counts(note_line_ids, line_counts):
        """Count actual notes per phrase; padding has phrase ID zero."""
        line_ids = torch.arange(1, int(line_counts.max()) + 1, device=note_line_ids.device)
        return note_line_ids[:, None].eq(line_ids[None, :, None]).sum(-1)

    def validate_counts(self, counts, line_counts):
        if counts.dtype not in (torch.int32, torch.int64) or counts.shape != (
            len(line_counts),
            int(line_counts.max()),
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
        """Fixed line prefixes and known slots within the song."""
        rows = []
        for row in counts.tolist():
            tokens, lines, slots, sizes = [], [], [], []
            for index, count in enumerate(row):
                if not count:
                    continue
                tokens.extend([self.first_line + index] + [SLOT] * count)
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

    def classify(self, hidden):
        strength = self.strength_head(hidden)
        length = self.length_head(hidden)
        return SimpleNamespace(
            logits=(strength.unsqueeze(-1) + length.unsqueeze(-2)).flatten(-2),
            strength_logits=strength,
            length_logits=length,
            role_logits=self.role_head(hidden) if self.role_head is not None else None,
        )

    def predict_notes(self, **source):
        return self.classify(self.head_dropout(self.encode(**source)))

    def align_note_predictions(
        self,
        notes,
        target_note_indices,
        syllable_counts,
        line_counts,
        note_line_ids,
        melody_attention_mask,
    ):
        """Gather observed note anchors into an IPA skeleton; never learn alignment."""
        self.validate_counts(syllable_counts, line_counts)
        indices = target_note_indices
        if indices.ndim != 2 or indices.dtype != torch.long or len(indices) != len(line_counts):
            raise ValueError("Invalid target_note_indices shape/type")
        skeleton = self.make_skeleton(syllable_counts, width=indices.shape[1])
        slots = skeleton["tokens"].eq(SLOT)
        active = indices.ge(0)
        if (
            (indices < -1).any()
            or (indices >= notes.logits.shape[1]).any()
            or (active & ~slots).any()
        ):
            raise ValueError("Invalid note index or mapped line prefix/padding")
        safe = indices.clamp_min(0)
        if (
            active
            & (
                ~melody_attention_mask.gather(1, safe).bool()
                | note_line_ids.gather(1, safe).ne(skeleton["line_ids"])
            )
        ).any():
            raise ValueError("Note alignment points to padding or another phrase")
        for row in indices:
            selected = row[row.ge(0)]
            if (selected[1:] <= selected[:-1]).any():
                raise ValueError("Note anchors must be unique and ordered")
        gathered = {}
        for name in ("logits", "strength_logits", "length_logits"):
            values = getattr(notes, name)
            gathered[name] = values.gather(
                1, safe[..., None].expand(-1, -1, values.shape[-1])
            ).masked_fill(~active[..., None], 0)
        return SimpleNamespace(**gathered, skeleton=skeleton, target_mask=active & slots)

    def forward(
        self,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        labels,
        syllable_counts,
        target_note_indices,
        meter_features=None,
        meter_mask=None,
        timing_residuals=None,
        note_roles=None,
    ):
        if (
            labels.ndim != 2
            or labels.shape[0] != len(melody_features)
            or labels.shape[1] > self.max_target_length
        ):
            raise ValueError("Invalid bridge target shape/length")
        if target_note_indices.shape != labels.shape:
            raise ValueError("Note-head training requires target_note_indices for the IPA skeleton")
        notes = self.predict_notes(
            melody_features=melody_features,
            melody_attention_mask=melody_attention_mask,
            note_line_ids=note_line_ids,
            line_counts=line_counts,
            meter_features=meter_features,
            meter_mask=meter_mask,
            timing_residuals=timing_residuals,
        )
        output = self.align_note_predictions(
            notes,
            target_note_indices,
            syllable_counts,
            line_counts,
            note_line_ids,
            melody_attention_mask,
        )
        skeleton = output.skeleton
        slots = skeleton["tokens"].eq(SLOT)
        structure = skeleton["tokens"].masked_fill(skeleton["tokens"].eq(PAD), -100)
        if (
            labels[~slots].ne(structure[~slots]).any()
            or ((labels[slots] < PAIR_OFFSET) | (labels[slots] >= self.first_line)).any()
        ):
            raise ValueError("Prosody labels must fit the supplied line skeleton")
        active = output.target_mask
        targets = labels[active] - PAIR_OFFSET
        components = {
            "strength": F.cross_entropy(output.strength_logits[active].float(), targets // 2)
            if active.any()
            else output.strength_logits.sum() * 0,
            "length": F.cross_entropy(output.length_logits[active].float(), targets % 2)
            if active.any()
            else output.length_logits.sum() * 0,
        }
        output.note_predictions = notes
        output.role_logits = notes.role_logits
        output.loss = sum(components.values())
        if self.predict_note_roles:
            if note_roles is None or note_roles.shape != melody_attention_mask.shape:
                raise ValueError("Role-head training requires per-note role targets")
            role_active = note_roles.ne(-100) & melody_attention_mask.bool()
            if ((note_roles[role_active] < 0) | (note_roles[role_active] > 1)).any():
                raise ValueError("Note roles must be START=0 or CONTINUE=1")
            components["role"] = (
                F.cross_entropy(notes.role_logits[role_active].float(), note_roles[role_active])
                if role_active.any()
                else notes.role_logits.sum() * 0
            )
            output.loss = output.loss + self.role_loss_weight * components["role"]
        output.loss_components = components
        return output

    def predicted_starts(self, notes, attention_mask, note_line_ids):
        active = attention_mask.bool()
        starts = (
            notes.role_logits.float().softmax(-1)[..., 1] < self.role_threshold
            if self.predict_note_roles
            else active.clone()
        )
        first = torch.ones_like(active)
        first[:, 1:] = note_line_ids[:, 1:].ne(note_line_ids[:, :-1])
        return (starts | first) & active

    def generate_from_notes(self, notes, melody_attention_mask, note_line_ids, line_counts):
        starts = self.predicted_starts(notes, melody_attention_mask, note_line_ids)
        counts = self.note_counts(note_line_ids.masked_fill(~starts, 0), line_counts)
        self.validate_counts(counts, line_counts)
        skeleton = self.make_skeleton(counts)
        indices = torch.full_like(skeleton["tokens"], -1)
        for row in range(len(counts)):
            indices[row, skeleton["tokens"][row].eq(SLOT)] = starts[row].nonzero().flatten()
        output = self.align_note_predictions(
            notes, indices, counts, line_counts, note_line_ids, melody_attention_mask
        )
        return SimpleNamespace(
            sequences=self.template_sequences(output),
            syllable_counts=counts,
            note_start_mask=starts,
            target_note_indices=indices,
        )

    def template_sequences(self, output):
        """Serialize parallel head predictions without feeding labels back to the model."""
        sequences = output.skeleton["tokens"].clone()
        slots = sequences.eq(SLOT)
        pairs = 2 * output.strength_logits.argmax(-1) + output.length_logits.argmax(-1)
        sequences[slots] = PAIR_OFFSET + pairs[slots]
        return sequences

    @torch.inference_mode()
    def generate(
        self,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        meter_features=None,
        meter_mask=None,
        timing_residuals=None,
        *,
        syllable_counts=None,
        target_note_indices=None,
    ):
        """Emit pairs at predicted starts (or every note); IPA anchors are scoring-only."""
        notes = self.predict_notes(
            melody_features=melody_features,
            melody_attention_mask=melody_attention_mask,
            note_line_ids=note_line_ids,
            line_counts=line_counts,
            meter_features=meter_features,
            meter_mask=meter_mask,
            timing_residuals=timing_residuals,
        )
        counts = self.note_counts(note_line_ids, line_counts)
        if target_note_indices is None:
            if syllable_counts is not None and not torch.equal(syllable_counts, counts):
                raise ValueError("IPA-count generation requires observed target_note_indices")
            return self.generate_from_notes(
                notes, melody_attention_mask, note_line_ids, line_counts
            )
        else:
            if syllable_counts is None:
                raise ValueError("Observed note alignment requires IPA syllable_counts")
            counts = syllable_counts
        output = self.align_note_predictions(
            notes,
            target_note_indices,
            counts,
            line_counts,
            note_line_ids,
            melody_attention_mask,
        )
        return SimpleNamespace(sequences=self.template_sequences(output), syllable_counts=counts)

    def configuration(self):
        return dict(
            melody_config=self.melody_config,
            max_syllables=self.max_syllables,
            max_song_notes=self.max_song_notes,
            max_song_lines=self.max_song_lines,
            max_target_length=self.max_target_length,
            strength_labels=list(self.strength_labels),
            alignment_policy=self.alignment_policy,
            meter_conditioning=self.meter_conditioning,
            contrastive_config=self.contrastive_config,
            representation=self.representation,
            octuple_config=self.octuple_config,
            predict_note_roles=self.predict_note_roles,
            role_loss_weight=self.role_loss_weight,
            role_threshold=self.role_threshold,
            role_class_weights=self.role_class_weights.tolist()
            if self.predict_note_roles
            else None,
            canonical_timing=self.canonical_timing,
        )

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            format_version=11
            if self.canonical_timing or self.predict_note_roles
            else FORMAT_VERSION,
            source_scheme=self.source_scheme,
            target_scheme=TARGET_SCHEME,
            melody_representation=self.representation,
            meter_feature_scheme=METER_FEATURE_SCHEME,
            alignment_scheme=ALIGNMENT_FORMAT,
            prosody_rules=self.prosody_rules,
            vocabulary=self.vocabulary,
            configuration=self.configuration(),
            provenance=self.provenance,
            supervised_exposure=self.supervised_exposure,
        )
        if self.representation == REPRESENTATION:
            metadata["octuple_codec"] = codec_metadata(self.octuple_config)
        (directory / "bridge.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save(self.state_dict(), directory / "bridge_weights.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "bridge.json").read_text(encoding="utf-8"))
        if meta.get("format_version") not in (FORMAT_VERSION, 11):
            raise ValueError(
                "Global bridge requires format v10; retrain older window/query bridges"
            )
        representation = meta.get("melody_representation")
        if representation not in (MELODY_REPRESENTATION, REPRESENTATION):
            raise ValueError("Unsupported checkpoint melody representation")
        if (
            meta.get("configuration", {}).get("representation", MELODY_REPRESENTATION)
            != representation
        ):
            raise ValueError("Checkpoint representation metadata conflict")
        if representation == REPRESENTATION and (
            meta.get("format_version") != 11
            or meta.get("octuple_codec")
            != codec_metadata(meta["configuration"].get("octuple_config"))
        ):
            raise ValueError("Incompatible octuple vocabulary, quantizer, or grid policy")
        if any(
            meta.get(key) != value
            for key, value in dict(
                source_scheme=(
                    REPRESENTATION
                    if representation == REPRESENTATION
                    else "global_note_177d_canonical_meter_v1"
                    if meta["configuration"].get("canonical_timing")
                    else SOURCE_SCHEME
                ),
                target_scheme=TARGET_SCHEME,
                melody_representation=representation,
                meter_feature_scheme=METER_FEATURE_SCHEME,
            ).items()
        ) or meta.get("alignment_scheme") not in (ALIGNMENT_FORMAT, ALIGNMENT_SCHEME):
            raise ValueError("Incompatible bridge checkpoint feature rules")
        model = cls(**meta["configuration"], provenance=meta["provenance"])
        model.supervised_exposure = meta.get("supervised_exposure")
        if (
            meta.get("prosody_rules") != model.prosody_rules
            or meta.get("vocabulary") != model.vocabulary
        ):
            raise ValueError("Incompatible bridge checkpoint feature rules or vocabulary")
        model.load_state_dict(
            torch.load(directory / "bridge_weights.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        return model
