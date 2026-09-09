"""BART with the original compound stress/length/remainder conditioning idea."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration, GenerationConfig


class ProsodyBart(nn.Module):
    def __init__(self, bart, *, max_syllables, dropout=0.1):
        super().__init__()
        self.bart = bart
        self.max_syllables = max_syllables
        self.dropout_probability = dropout
        hidden_size = bart.config.d_model
        self.length_embedding = nn.Embedding(3, hidden_size, padding_idx=0)
        self.remainder_embedding = nn.Embedding(max_syllables + 1, hidden_size, padding_idx=0)
        self.projection = nn.Linear(3 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def embed_source(self, input_ids, length_ids, remainder_ids):
        words = self.bart.get_encoder().embed_tokens(input_ids)
        lengths = self.length_embedding(length_ids)
        remainders = self.remainder_embedding(remainder_ids)
        return self.dropout(self.projection(torch.cat((words, lengths, remainders), dim=-1)))

    def forward(self, input_ids, length_ids, remainder_ids, attention_mask, labels):
        return self.bart(
            inputs_embeds=self.embed_source(input_ids, length_ids, remainder_ids),
            attention_mask=attention_mask,
            labels=labels,
            # Labels are shifted by BART. Never pass unshifted target embeddings.
            use_cache=False,
        )

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
        metadata = {"max_syllables": self.max_syllables, "dropout": self.dropout_probability}
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
