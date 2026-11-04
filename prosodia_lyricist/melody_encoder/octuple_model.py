"""Compound field embeddings using the shared whole-song Transformer."""

import math

import torch
from torch import nn

from .modeling import MelodyTransformerEncoder
from .octuple import OFFSET, codec_config, time_centers


class OctupleEmbedding(nn.Module):
    def __init__(self, width, resolution):
        super().__init__()
        self.resolution = resolution
        self.bar_specials = nn.Embedding(OFFSET, 32, padding_idx=0)
        self.quarter = nn.Embedding(OFFSET + 1024, 16, padding_idx=0)
        self.subdivision = nn.Embedding(OFFSET + resolution, 16, padding_idx=0)
        self.numerator = nn.Embedding(OFFSET + 255, 16, padding_idx=0)
        self.denominator = nn.Embedding(OFFSET + 7, 16, padding_idx=0)
        self.fields = nn.ModuleDict(
            {
                key: nn.Embedding(OFFSET + size, 32, padding_idx=0)
                for key, size in dict(
                    pitch=128,
                    interval=255,
                    duration=len(time_centers(resolution)) + 1,
                    gap=len(time_centers(resolution)) + 1,
                    tempo=147,
                ).items()
            }
        )
        self.project = nn.Sequential(nn.Linear(256, width), nn.LayerNorm(width))
        self.register_buffer(
            "frequencies",
            torch.exp(torch.arange(0, 32, 2).float() * (-math.log(10000.0) / 32)),
            persistent=False,
        )

    @staticmethod
    def factor(ids, divisor):
        raw = (ids - OFFSET).clamp_min(0)
        return (
            torch.where(ids < OFFSET, ids, raw // divisor + OFFSET),
            torch.where(ids < OFFSET, ids, raw % divisor + OFFSET),
        )

    def forward(self, tokens):
        if tokens.dtype != torch.long or tokens.shape[-1] != 8 or (tokens < 0).any():
            raise ValueError("Octuple tokens must be nonnegative int64 [batch, notes, 8]")
        bar = tokens[..., 0]
        phase = (bar - OFFSET).clamp_min(0).float()[..., None] * self.frequencies
        bar_vector = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        bar_vector = torch.where(
            (bar < OFFSET)[..., None], self.bar_specials(bar.clamp_max(OFFSET - 1)), bar_vector
        )
        quarter, subdivision = self.factor(tokens[..., 1], self.resolution)
        num, den = self.factor(tokens[..., 6], 7)
        vectors = [
            bar_vector,
            torch.cat((self.quarter(quarter), self.subdivision(subdivision)), -1),
        ]
        for index, key in ((2, "pitch"), (3, "interval"), (4, "duration"), (5, "gap")):
            vectors.append(self.fields[key](tokens[..., index]))
        vectors.extend(
            (
                torch.cat((self.numerator(num), self.denominator(den)), -1),
                self.fields["tempo"](tokens[..., 7]),
            )
        )
        return self.project(torch.cat(vectors, -1))


class OctupleMelodyEncoder(MelodyTransformerEncoder):
    def __init__(self, *, octuple_config=None, **kwargs):
        super().__init__(input_dim=8, **kwargs)
        self.octuple_config = codec_config(octuple_config)
        self.input_projection = OctupleEmbedding(self.d_model, self.octuple_config["resolution"])
        self.residual_projection = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, self.d_model)
        )

    def encode(
        self,
        melody_features,
        melody_attention_mask=None,
        *,
        timing_residuals=None,
        note_context=None,
        **kwargs,
    ):
        if timing_residuals is None or timing_residuals.shape != (*melody_features.shape[:2], 3):
            raise ValueError("Octuple input requires three timing residuals per note")
        if not torch.isfinite(timing_residuals).all():
            raise ValueError("Timing residuals must be finite")
        if self.octuple_config["timing_residuals"]:
            residual = self.residual_projection(timing_residuals)
            note_context = residual if note_context is None else note_context + residual
        return super().encode(
            melody_features, melody_attention_mask, note_context=note_context, **kwargs
        )
