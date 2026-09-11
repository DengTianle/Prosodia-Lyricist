"""BART with the original compound stress/length/remainder conditioning idea."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration, GenerationConfig


class ProsodyBart(nn.Module):
    def __init__(self, bart, *, max_syllables, dropout=0.1, loss_weights=None):
        super().__init__()
        self.bart = bart
        self.max_syllables = max_syllables
        self.dropout_probability = dropout
        hidden_size = bart.config.d_model
        self.length_embedding = nn.Embedding(3, hidden_size, padding_idx=0)
        self.remainder_embedding = nn.Embedding(max_syllables + 1, hidden_size, padding_idx=0)
        self.projection = nn.Linear(3 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        for embedding in (self.length_embedding, self.remainder_embedding):
            nn.init.normal_(embedding.weight, mean=0, std=hidden_size**-0.5)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.loss_weights = {"word": 1.0, "syllable": 0.0, "remainder": 0.0, "sentence": 0.0}
        if loss_weights:
            if set(loss_weights) - self.loss_weights.keys():
                raise ValueError("Unknown auxiliary loss name")
            self.loss_weights.update(loss_weights)
        if any(not 0 <= v < float("inf") for v in self.loss_weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not any(self.loss_weights.values()):
            raise ValueError("At least one loss must be enabled")
        self.auxiliary_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_size, 2 if name == "sentence" else max_syllables + 1)
                for name in ("syllable", "remainder", "sentence")
                if self.loss_weights[name] > 0
            }
        )

    def embed_source(self, input_ids, length_ids, remainder_ids):
        words = self.bart.get_encoder().embed_tokens(input_ids)
        lengths = self.length_embedding(length_ids)
        remainders = self.remainder_embedding(remainder_ids)
        return self.dropout(self.projection(torch.cat((words, lengths, remainders), dim=-1)))

    def forward(
        self,
        input_ids,
        length_ids,
        remainder_ids,
        attention_mask,
        labels,
        syllable_labels=None,
        remainder_labels=None,
        sentence_labels=None,
    ):
        output = self.bart(
            inputs_embeds=self.embed_source(input_ids, length_ids, remainder_ids),
            attention_mask=attention_mask,
            labels=labels,
            # Labels are shifted by BART. Never pass unshifted target embeddings.
            use_cache=False,
            output_hidden_states=bool(self.auxiliary_heads),
        )
        losses = {"word": output.loss}
        targets = {
            "syllable": syllable_labels,
            "remainder": remainder_labels,
            "sentence": sentence_labels,
        }
        total = self.loss_weights["word"] * output.loss
        for name, head in self.auxiliary_heads.items():
            target = targets[name]
            if target is None or target.shape != labels.shape or not target.ne(-100).any():
                raise ValueError(f"Enabled {name} loss requires aligned, nonempty scaffold labels")
            logits = head(output.decoder_hidden_states[-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100
            )
            losses[name] = loss
            total = total + self.loss_weights[name] * loss
        output.loss = total
        output["loss_components"] = losses
        return output

    @torch.no_grad()
    def generate(self, input_ids, length_ids, remainder_ids, attention_mask, **kwargs):
        return self.bart.generate(
            inputs_embeds=self.embed_source(input_ids, length_ids, remainder_ids),
            attention_mask=attention_mask,
            **kwargs,
        )

    def save(self, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bart.config.save_pretrained(directory)
        self.bart.generation_config.save_pretrained(directory)
        tokenizer.save_pretrained(directory / "tokenizer")
        metadata = {
            "max_syllables": self.max_syllables,
            "dropout": self.dropout_probability,
            "loss_weights": self.loss_weights,
        }
        (directory / "prosody.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Replace the weights atomically so interrupted saves do not corrupt the last checkpoint.
        temporary = directory / "weights.pt.tmp"
        torch.save(self.state_dict(), temporary)
        temporary.replace(directory / "weights.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = BartConfig.from_pretrained(directory, local_files_only=True)
        metadata = json.loads((directory / "prosody.json").read_text(encoding="utf-8"))
        model = cls(BartForConditionalGeneration(config), **metadata)
        model.bart.generation_config = GenerationConfig.from_pretrained(
            directory, local_files_only=True
        )
        model.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True), strict=True
        )
        return model
