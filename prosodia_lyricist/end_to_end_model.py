"""Parallel melody conditioning for the unchanged four-stream ProsodyBart decoder."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .bridge_data import PAIR_OFFSET, SLOT
from .bridge_model import ProsodyBridge
from .end_to_end_data import CONDITION_FIELDS
from .model import LOSS_NAMES, ProsodyBart


class EndToEndProsody(nn.Module):
    def __init__(
        self,
        bridge,
        template_model,
        *,
        stress_token_ids,
        temperature=1.0,
        anchor_weight=0.0,
        template_trainable="frozen",
        melody_trainable=False,
        loss_weights=None,
        provenance=None,
    ):
        super().__init__()
        if bridge.query_mode != "parallel" or not isinstance(template_model, ProsodyBart):
            raise ValueError("Expected a parallel bridge and an explainable template decoder")
        if bridge.max_syllables != template_model.max_syllables:
            raise ValueError("Bridge and template decoder syllable vocabularies differ")
        if not 0 < temperature < math.inf or not 0 <= anchor_weight < math.inf:
            raise ValueError("Temperature must be positive; anchor weight must be nonnegative")
        if (
            len(stress_token_ids) != 2
            or len(set(stress_token_ids)) != 2
            or any(
                not isinstance(i, int) or not 0 <= i < template_model.bart.config.vocab_size
                for i in stress_token_ids
            )
        ):
            raise ValueError("Expected vocabulary IDs for strong and weak source tokens")
        weights = {**template_model.loss_weights, **(loss_weights or {})}
        if (
            set(weights) != set(LOSS_NAMES)
            or any(not 0 <= w < math.inf for w in weights.values())
            or weights["lyrics"] <= 0
        ):
            raise ValueError("Joint training requires positive lyric loss and valid loss weights")
        template_model.loss_weights = weights
        self.bridge = bridge
        self.template_model = template_model
        self.temperature, self.anchor_weight = temperature, anchor_weight
        self.provenance = provenance or {}
        self.register_buffer("stress_token_ids", torch.tensor(stress_token_ids), persistent=False)
        self.set_template_trainable(template_trainable)
        self.bridge.set_melody_trainable(melody_trainable)

    def set_template_trainable(self, mode):
        if mode not in ("frozen", "encoder", "all"):
            raise ValueError("template_trainable must be frozen, encoder, or all")
        self.template_trainable = mode
        self.template_model.requires_grad_(mode == "all")
        if mode == "encoder":
            encoder = self.template_model.bart.get_encoder()
            # Do not unfreeze encoder.embed_tokens: BART shares it with the decoder
            # and LM head. The length embedding is also shared with target inputs.
            for module in (
                encoder.layers,
                encoder.embed_positions,
                encoder.layernorm_embedding,
                self.template_model.projection,
            ):
                module.requires_grad_(True)
        self._template_mode()

    def _template_mode(self):
        self.template_model.train(self.training and self.template_trainable == "all")
        if self.template_trainable == "encoder":
            self.template_model.bart.get_encoder().train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self._template_mode()
        return self

    @property
    def dataset_options(self):
        bridge = self.bridge
        return dict(
            max_syllables=bridge.max_syllables,
            encoder_lines_per_window=bridge.encoder_lines_per_window,
            max_window_notes=bridge.max_notes,
            max_song_notes=bridge.max_song_notes,
            max_song_lines=bridge.max_song_lines,
            max_bridge_length=bridge.max_target_length,
        )

    def condition_source(
        self,
        input_ids,
        length_ids,
        attention_mask,
        slot_indices,
        syllable_counts,
        melody_features,
        melody_attention_mask,
        note_line_ids,
        line_counts,
        song_features,
        *,
        template_mode="soft",
    ):
        if template_mode not in ("soft", "hard"):
            raise ValueError("template_mode must be soft or hard")
        if (
            input_ids.shape != slot_indices.shape
            or input_ids.shape != attention_mask.shape
            or (input_ids.shape != length_ids.shape or slot_indices.dtype != torch.long)
        ):
            raise ValueError("Source IDs, masks, and slot indices must align")
        if input_ids.shape[1] > self.template_model.bart.config.max_position_embeddings:
            raise ValueError("Source exceeds the template encoder position limit")
        output = self.bridge.predict_slots(
            melody_features,
            melody_attention_mask,
            note_line_ids,
            line_counts,
            syllable_counts,
            song_features,
        )
        slots = slot_indices.ge(0)
        if (
            (slots & ~attention_mask.bool()).any()
            or (slot_indices.ge(output.logits.shape[1]).any())
            or not torch.equal(slots.sum(-1), syllable_counts.sum(-1))
        ):
            raise ValueError("Source slots do not match the bridge skeleton")
        for row in range(len(input_ids)):
            expected = output.skeleton["tokens"][row].eq(SLOT).nonzero().flatten()
            if not torch.equal(slot_indices[row, slots[row]], expected):
                raise ValueError("Source slot indices must cover the ordered bridge slots once")
        probabilities = [
            (logits.float() / self.temperature).softmax(-1)
            for logits in (output.strength_logits, output.length_logits)
        ]
        if template_mode == "hard":
            probabilities = [F.one_hot(p.argmax(-1), 2).to(p.dtype) for p in probabilities]
        indices = slot_indices.clamp_min(0)[..., None].expand(-1, -1, 2)
        stress, length = [p.gather(1, indices) for p in probabilities]
        model = self.template_model
        # Calling the embedding module preserves BART's configured embedding scale.
        stress_bank = model.bart.get_encoder().embed_tokens(self.stress_token_ids)
        length_bank = model.length_embedding.weight[1:3]  # 1=long, 2=short; never pad.
        mixed = model.dropout(
            model.projection(
                torch.cat(
                    (
                        stress.to(stress_bank.dtype) @ stress_bank,
                        length.to(length_bank.dtype) @ length_bank,
                    ),
                    dim=-1,
                )
            )
        )
        ordinary = model.embed_source(input_ids, length_ids)
        embeddings = torch.where(slots[..., None], mixed, ordinary)
        sequences = output.skeleton["tokens"].clone()
        active = sequences.eq(SLOT)
        pairs = 2 * output.strength_logits.argmax(-1) + output.length_logits.argmax(-1)
        sequences[active] = PAIR_OFFSET + pairs[active]
        return SimpleNamespace(
            inputs_embeds=embeddings,
            bridge=output,
            sequences=sequences,
            stress_probabilities=probabilities[0],
            length_probabilities=probabilities[1],
        )

    def forward(
        self,
        *,
        labels,
        syllable_labels,
        stress_labels,
        length_labels,
        bridge_labels=None,
        template_mode="soft",
        **condition,
    ):
        source = self.condition_source(**condition, template_mode=template_mode)
        # Frozen weights still participate in autograd with respect to source embeddings.
        # Target words AND all three shifted target-feature streams are unchanged.
        output = self.template_model(
            input_ids=condition["input_ids"],
            length_ids=condition["length_ids"],
            attention_mask=condition["attention_mask"],
            inputs_embeds=source.inputs_embeds,
            labels=labels,
            syllable_labels=syllable_labels,
            stress_labels=stress_labels,
            length_labels=length_labels,
        )
        if bridge_labels is not None:
            if bridge_labels.shape != source.bridge.logits.shape[:2]:
                raise ValueError("Bridge anchor labels must align with structural queries")
            active = bridge_labels.ne(-100)
            if (active & source.bridge.skeleton["tokens"].ne(SLOT)).any() or (
                ((bridge_labels[active] < 0) | (bridge_labels[active] > 3)).any()
            ):
                raise ValueError("Invalid bridge anchor labels")
            anchor = (
                F.cross_entropy(source.bridge.strength_logits[active], bridge_labels[active] // 2)
                + F.cross_entropy(source.bridge.length_logits[active], bridge_labels[active] % 2)
                if active.any()
                else source.bridge.logits.sum() * 0
            )
            output["loss_components"] = {**output.loss_components, "bridge_anchor": anchor}
            output.loss = output.loss + self.anchor_weight * anchor
        elif self.anchor_weight:
            raise ValueError("Bridge anchor supervision is enabled but labels are missing")
        output["bridge_output"] = source.bridge
        return output

    @torch.inference_mode()
    def generate(self, *, tokenizer, template_mode="soft", **kwargs):
        if self.training:
            raise ValueError("Call model.eval() before generation")
        condition = {key: kwargs.pop(key) for key in CONDITION_FIELDS}
        source = self.condition_source(**condition, template_mode=template_mode)
        result = self.template_model.generate(
            input_ids=condition["input_ids"],
            length_ids=condition["length_ids"],
            attention_mask=condition["attention_mask"],
            inputs_embeds=source.inputs_embeds,
            tokenizer=tokenizer,
            **kwargs,
        )
        # These are diagnostics; generation used soft embeddings unless explicitly ablated.
        for row, explanation in enumerate(result.explanations):
            slots = source.bridge.skeleton["tokens"][row].eq(SLOT)
            explanation["melody_conditioning"] = {
                "mode": template_mode,
                "temperature": self.temperature,
                "hard_template_tokens": source.sequences[row].tolist(),
                "stress_probabilities": source.stress_probabilities[row, slots].tolist(),
                "length_probabilities": source.length_probabilities[row, slots].tolist(),
            }
        return result

    def save(self, directory, tokenizer):
        directory = Path(directory)
        self.template_model.save(directory, tokenizer)
        self.bridge.save(directory / "bridge")
        metadata = dict(
            format_version=1,
            conditioning="parallel_soft_template",
            stress_token_ids=self.stress_token_ids.tolist(),
            temperature=self.temperature,
            anchor_weight=self.anchor_weight,
            template_trainable=self.template_trainable,
            melody_trainable=not self.bridge.melody_frozen,
            provenance=self.provenance,
        )
        (directory / "end_to_end.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "end_to_end.json").read_text(encoding="utf-8"))
        if meta.pop("format_version") != 1 or meta.pop("conditioning") != "parallel_soft_template":
            raise ValueError("Unsupported end-to-end checkpoint format")
        return cls(ProsodyBridge.load(directory / "bridge"), ProsodyBart.load(directory), **meta)


def condition_inputs(batch):
    """Explicit allowlist: decoder targets and anchor labels can never enter the bridge."""
    return {key: batch[key] for key in CONDITION_FIELDS}
