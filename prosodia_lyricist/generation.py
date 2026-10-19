"""Word-complete sampling with IPA correction before subsequent decoder steps."""

import math
import re
from dataclasses import dataclass

import torch

from . import ipa
from .features import WORD_END, token_id, word_prosody


@dataclass
class TemplateGeneration:
    # Each row includes decoder-start, BOS, and the generated events. Feature zero
    # is the paper's non-word <pad>; rectangular batch padding also has mask zero.
    sequences: torch.Tensor
    syllable_ids: torch.Tensor
    stress_ids: torch.Tensor
    length_ids: torch.Tensor
    attention_mask: torch.Tensor
    explanations: list


def sample(logits, *, do_sample, temperature, top_k, allowed=None):
    logits = logits.clone() / temperature
    if allowed is not None:
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[allowed] = False
        logits.masked_fill_(mask, -torch.inf)
    if not torch.isfinite(logits).any():
        raise ValueError("No valid next symbol under the decoding constraints")
    if not do_sample:
        return int(logits.argmax())
    values, indices = logits.topk(min(top_k, int(torch.isfinite(logits).sum())))
    return int(indices[torch.multinomial(values.softmax(-1), 1)])


def resolve_repetition_settings(model, no_repeat_ngram_size=None, repetition_penalty=None):
    config = model.bart.generation_config
    if no_repeat_ngram_size is None:
        no_repeat_ngram_size = config.no_repeat_ngram_size
    if repetition_penalty is None:
        repetition_penalty = config.repetition_penalty
    if (
        isinstance(no_repeat_ngram_size, bool)
        or not isinstance(no_repeat_ngram_size, int)
        or no_repeat_ngram_size < 0
    ):
        raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
    if (
        isinstance(repetition_penalty, bool)
        or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(repetition_penalty)
        or repetition_penalty <= 0
    ):
        raise ValueError("repetition_penalty must be positive and finite")
    return {
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
    }


def apply_text_repetition(logits, history, *, no_repeat_ngram_size, repetition_penalty):
    """Adjust lyric BPE logits; history excludes all structural/special events."""
    scores = logits.clone()
    if repetition_penalty != 1 and history:
        ids = sorted(set(history))
        values = scores[ids]
        scores[ids] = torch.where(
            values < 0, values * repetition_penalty, values / repetition_penalty
        )
    size = no_repeat_ngram_size
    if size and len(history) >= size - 1:
        prefix = history[-(size - 1):] if size > 1 else []
        banned = [
            history[i + size - 1]
            for i in range(len(history) - size + 1)
            if history[i:i + size - 1] == prefix
        ]
        scores[banned] = -torch.inf
    return scores


def generate_templates(
    model,
    tokenizer,
    input_ids,
    length_ids,
    attention_mask,
    *,
    max_new_tokens=256,
    do_sample=False,
    temperature=1.0,
    top_k=3,
    prosody_correction=True,
    pronunciation=None,
    inputs_embeds=None,
    no_repeat_ngram_size=None,
    repetition_penalty=None,
):
    """Generate one independent song per batch row, preserving four aligned streams.

    BART's BPE vocabulary needs an explicit completed-word event. Its three labels
    are sampled from separate heads, then replaced with IPA-derived labels. Only
    the corrected event enters the decoder cache, so no stale cache replay occurs.
    Unknown pronunciations fail explicitly when correction is enabled; there is
    no silent substitution of uncorrected labels for supposedly corrected output.
    """
    if max_new_tokens < 1 or max_new_tokens >= model.bart.config.max_position_embeddings:
        raise ValueError("max_new_tokens must fit the decoder position limit (including start)")
    if temperature <= 0 or top_k < 1:
        raise ValueError("temperature and top_k must be positive")
    if model.training:
        raise ValueError("Call model.eval() before generation")
    repetition = resolve_repetition_settings(model, no_repeat_ngram_size, repetition_penalty)
    pronunciation = pronunciation or ipa.parse_words
    end_word = token_id(tokenizer, WORD_END)
    eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id
    special_ids = set(tokenizer.all_special_ids)
    forbidden = special_ids - {end_word, eos}
    # Token text comes from decoding the real tokenizer, not BPE spelling guesses.
    text_by_id = [
        tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))
    ]
    lexical = {i for i, text in enumerate(text_by_id) if any(c.isalpha() for c in text)}
    # Standalone whitespace tokens also delimit words in byte-level BPE.
    starts_word = {i for i, text in enumerate(text_by_id) if any(c.isspace() for c in text)}
    punctuation = {
        i
        for i, text in enumerate(text_by_id)
        if text.strip() and not any(c.isalnum() for c in text) and text.strip() not in ("'", "’")
    }
    word_level = type(tokenizer.backend_tokenizer.model).__name__ == "WordLevel"
    all_ids = set(range(len(tokenizer))) - forbidden
    encoder = model.bart.get_encoder()(
        inputs_embeds=(
            model.embed_source(input_ids, length_ids) if inputs_embeds is None else inputs_embeds
        ),
        attention_mask=attention_mask,
        return_dict=True,
    )
    from transformers.modeling_outputs import BaseModelOutput

    results, explanations = [], []
    for row in range(input_ids.shape[0]):
        streams = [[model.bart.config.decoder_start_token_id], [0], [0], [0]]
        past = None
        pending = []
        pending_start = None
        words = []
        text_history = []
        complete = False
        for step in range(max_new_tokens):
            last = [torch.tensor([[s[-1]]], device=input_ids.device) for s in streams]
            output = model.bart(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoder.last_hidden_state[row : row + 1]
                ),
                attention_mask=attention_mask[row : row + 1],
                decoder_inputs_embeds=model.embed_target(*last),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = output.past_key_values
            allowed = all_ids.copy()
            if step == 0:
                # BOS is the first teacher-forced target in training.
                allowed = {bos}
            elif pending:
                allowed.discard(eos)
                allowed -= starts_word | punctuation
                allowed.add(end_word)
                if word_level or len(pending) >= 24:
                    allowed = {end_word}
            else:
                allowed.discard(end_word)
            chosen = sample(
                apply_text_repetition(output.logits[0, -1], text_history, **repetition),
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                allowed=sorted(allowed),
            )
            if chosen not in special_ids:
                text_history.append(chosen)
            features = (0, 0, 0)
            if chosen == end_word:
                hidden = output.decoder_hidden_states[-1][0, -1]
                predicted = tuple(
                    sample(
                        head(hidden),
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        allowed=list(range(1, head.out_features)),
                    )
                    for head in model.prosody_heads.values()
                )
                word = tokenizer.decode(pending, clean_up_tokenization_spaces=False).strip()
                features = predicted
                syllables = None
                if prosody_correction:
                    parsed = pronunciation(word)
                    spoken = [entry for entry in parsed if entry["syllables"]]
                    if len(spoken) != 1:
                        raise ValueError(f"Expected one completed IPA word, got {word!r}")
                    syllables = spoken[0]["syllables"]
                    features = word_prosody(syllables)
                    if features[0] > model.max_syllables:
                        raise ValueError(
                            f"IPA word exceeds checkpoint syllable vocabulary: {word!r}"
                        )
                words.append(
                    {
                        "text": word,
                        "token_start": pending_start,
                        "token_end": len(streams[0]),
                        "predicted": dict(
                            zip(("syllables", "stress", "length"), predicted, strict=True)
                        ),
                        "corrected": dict(
                            zip(("syllables", "stress", "length"), features, strict=True)
                        ),
                        "ipa_syllables": syllables,
                        "correction_applied": features != predicted,
                    }
                )
                pending = []
                pending_start = None
            elif chosen not in tokenizer.all_special_ids and (chosen in lexical or pending):
                if not pending:
                    pending_start = len(streams[0])
                pending.append(chosen)
            for stream, value in zip(streams, (chosen, *features), strict=True):
                stream.append(value)
            if chosen == eos:
                complete = True
                break
        # Never present an unfinished BPE fragment as a corrected lyric word.
        unfinished = tokenizer.decode(pending) if pending else ""
        if pending:
            for stream in streams:
                del stream[pending_start:]
        # A boundary must remain a word separator even if the next sampled BPE
        # token has no leading space; otherwise displayed lyrics disagree with IPA.
        text = tokenizer.decode(
            streams[0][1:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).replace(WORD_END, " ")
        for special in tokenizer.all_special_tokens:
            text = text.replace(special, "")
        text = re.sub(r"\s+([.,!?;:])", r"\1", " ".join(text.split()))
        explanations.append(
            {
                "text": text,
                "lines": [line.strip() for line in text.split(".") if line.strip()] or [""],
                "words": words,
                "completed": complete,
                "generated_events": step + 1,
                "hit_token_limit": not complete,
                "truncated_word": unfinished,
                "prosody_correction": prosody_correction,
                "repetition_controls": repetition,
                "label_vocabulary": {
                    "stress": {0: "pad", 1: "strong", 2: "weak"},
                    "length": {0: "pad", 1: "long", 2: "short"},
                },
            }
        )
        results.append(streams)
    width = max(len(streams[0]) for streams in results)
    tensors = [
        torch.tensor(
            [
                streams[i] + [tokenizer.pad_token_id if i == 0 else 0] * (width - len(streams[i]))
                for streams in results
            ],
            device=input_ids.device,
        )
        for i in range(4)
    ]
    masks = torch.tensor(
        [[1] * len(streams[0]) + [0] * (width - len(streams[0])) for streams in results],
        device=input_ids.device,
    )
    return TemplateGeneration(*tensors, masks, explanations)
