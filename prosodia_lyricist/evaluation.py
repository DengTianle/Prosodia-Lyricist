"""Model-independent prosody-BLEU and token-weighted conditional perplexity.

These implement documented paper-level definitions, not the broken metric
snippets in the public reference checkout. See docs/evaluation.md.
"""

import math
import unicodedata
from collections import Counter

import torch
from torch.nn import functional as F

from . import ipa

# English diphthongs, including common rhotic/centering transcription variants.
DIPHTHONGS = ("eɪ", "aɪ", "ɔɪ", "oʊ", "əʊ", "aʊ", "ɪə", "eə", "ɛə", "ʊə")
METRIC_VERSION = "xai-paper-reconstruction-v1"


def paper_syllable(syllable):
    """Supplement Eqs. (6)-(7), independent of historical training labels."""
    value = unicodedata.normalize("NFC", syllable["ipa"])
    return {
        "ipa": syllable["ipa"],
        "stress": "strong" if any(c in value for c in ("'", "`", "ˈ", "ˌ")) else "weak",
        "length": "long"
        if ":" in value or "ː" in value or any(d in value for d in DIPHTHONGS)
        else "short",
    }


def text_prosody(text):
    if not text.strip():
        return []
    return [
        {**paper_syllable(syllable), "word": word["text"]}
        for word in ipa.parse_words(text)
        for syllable in word["syllables"]
    ]


def bleu(reference, hypothesis, *, order=4):
    """Single-reference, unsmoothed BLEU with fixed, equal 1..order weights.

    Atomic compound prosody symbols are supplied by the caller. No tokenizer,
    BOS/EOS, or cross-phrase n-grams. An absent n-gram order yields exact zero.
    """
    if order < 1:
        raise ValueError("BLEU order must be positive")
    if not hypothesis or not reference:
        return 0.0
    precisions = []
    for n in range(1, order + 1):
        refs = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        hyps = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        matches = sum((refs & hyps).values())
        total = sum(hyps.values())
        if not matches or not total:
            return 0.0
        precisions.append(matches / total)
    penalty = min(0.0, 1.0 - len(reference) / len(hypothesis))
    return math.exp(penalty + sum(math.log(p) for p in precisions) / order)


def prosody_bleu(reference, hypothesis):
    def symbols(syllables):
        return [(s["stress"], s["length"]) for s in syllables]

    return bleu(symbols(reference), symbols(hypothesis))


def evaluate_prosody(records, lyrics):
    """Pair by phrase order; missing/extra/unpronounceable phrases score zero."""
    phrases = []
    for index in range(max(len(records), len(lyrics))):
        source = records[index]["syllables"] if index < len(records) else []
        text = lyrics[index] if index < len(lyrics) else ""
        error = None
        try:
            generated = text_prosody(text) if text.strip() else []
        except ValueError as exc:
            # Do not silently drop bad samples. Environment/resource failures
            # (RuntimeError) still abort, so a broken installation is not scored.
            generated = None
            error = str(exc)
        supported = all(s["stress"] in ("strong", "weak") for s in source)
        score = prosody_bleu(source, generated or []) if supported else None
        phrases.append(
            {
                "phrase": index + 1,
                "input_syllables": source,
                "text": text,
                "generated_syllables": generated,
                "input_count": len(source),
                "generated_count": None if generated is None else len(generated),
                "prosody_bleu": score,
                "error": error or (None if supported else "Input stress is unknown"),
                "status": (
                    "extra" if not source else "missing" if index >= len(lyrics) else "paired"
                ),
            }
        )
    scores = [phrase["prosody_bleu"] for phrase in phrases]
    return {
        "version": METRIC_VERSION,
        "prosody_bleu": (
            sum(scores) / len(scores) if scores and all(s is not None for s in scores) else None
        ),
        "aggregation": "phrase macro mean; missing/extra/IPA-failed phrases count as zero",
        "input_phrases": len(records),
        "generated_phrases": len(lyrics),
        "pronunciation_failures": sum(p["generated_syllables"] is None for p in phrases),
        "phrases": phrases,
    }


def nll_statistics(logits, labels, *, bos_token_id, pad_token_id):
    """Sum raw-logit CE over scored target BPEs, including periods and EOS."""
    targets = labels.clone()
    targets[(targets == bos_token_id) | (targets == pad_token_id)] = -100
    count = int(targets.ne(-100).sum())
    if not count:
        raise ValueError("No target tokens to score")
    nll = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return {"nll_sum": float(nll), "token_count": count}


def aggregate_perplexity(statistics):
    """Aggregate sufficient statistics, never average batch perplexities."""
    count = sum(s["token_count"] for s in statistics)
    total = sum(s["nll_sum"] for s in statistics)
    if count <= 0 or not math.isfinite(total):
        raise ValueError("Perplexity requires finite NLL and scored tokens")
    mean = total / count
    return {
        "nll_sum": total,
        "token_count": count,
        "mean_nll": mean,
        "perplexity": math.exp(mean) if mean < 709 else None,
        "overflow": mean >= 709,
    }


@torch.inference_mode()
def conditional_perplexity(model, inputs, labels, tokenizer):
    # Call BART directly: auxiliary heads/loss weights must not affect perplexity.
    output = model.bart(
        inputs_embeds=model.embed_source(
            inputs["input_ids"], inputs["length_ids"], inputs["remainder_ids"]
        ),
        attention_mask=inputs["attention_mask"],
        labels=labels,
        use_cache=False,
    )
    return aggregate_perplexity(
        [
            nll_statistics(
                output.logits,
                labels,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        ]
    )
