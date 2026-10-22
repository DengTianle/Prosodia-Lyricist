"""The single-stream reference checkpoint's source format and causal lyric targets."""

import logging
from pathlib import Path

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
TARGET_KEYS = ("labels",)
BASE_VOCAB_SIZE = 50265
REFERENCE_VOCAB_SIZE = 50322
REFERENCE_TOKEN_IDS = {
    "<title>": 50316,
    "<keywords>": 50317,
    "<template>": 50318,
    "<strong>": 50319,
    "<substrong>": 50320,
    "<weak>": 50321,
}


STRESSES = ("strong", "substrong", "weak", "unknown")
LENGTH_IDS = {"long": 1, "short": 2}
SOURCE_KEYS = ("input_ids", "length_ids", "remainder_ids", "attention_mask")


def special_tokens(max_syllables):
    return (
        ["<title>", "<template>", "<keywords>"]
        + [f"<{stress}>" for stress in STRESSES]
        + [f"<syllable_{n}>" for n in range(1, max_syllables + 1)]
    )


def configure_tokenizer(tokenizer, max_syllables):
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens(max_syllables)})
    tokenizer.padding_side = "right"
    return tokenizer


def record_lines(record):
    # A single phrase is also a valid one-line song (e.g. a short MIDI).
    lines = record["lines"] if "lines" in record else [record]
    if not lines:
        raise ValueError("Expected at least one lyric line")
    return lines


def token_id(tokenizer, token):
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or value == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer is missing the template token {token}")
    return value


def encode_source(record, tokenizer, max_syllables):
    ids = [tokenizer.bos_token_id, token_id(tokenizer, "<title>")]
    ids += tokenizer.encode(record.get("title", "").replace(".", ""), add_special_tokens=False)
    lengths, remainders = [0] * len(ids), [0] * len(ids)

    def append_plain(tokens):
        ids.extend(tokens)
        lengths.extend([0] * len(tokens))
        remainders.extend([0] * len(tokens))

    for line in record_lines(record):
        syllables = line["syllables"]
        count = len(syllables)
        if not 1 <= count <= max_syllables:
            raise ValueError(f"Expected 1..{max_syllables} syllables per line, got {count}")
        append_plain(
            [token_id(tokenizer, f"<syllable_{count}>"), token_id(tokenizer, "<template>")]
        )
        for index, syllable in enumerate(syllables):
            ids.append(token_id(tokenizer, f"<{syllable['stress']}>"))
            lengths.append(LENGTH_IDS[syllable["length"]])
            # Remaining+1 reserves zero for padding; reset for each line.
            remainders.append(count - index)
        # The released XAI source omits keyword markers and text.
        append_plain(tokenizer.encode(".", add_special_tokens=False))
    append_plain([tokenizer.eos_token_id])
    return {
        "input_ids": ids,
        "length_ids": lengths,
        "remainder_ids": remainders,
        "attention_mask": [1] * len(ids),
    }


def load_tokenizer(directory=None, *, cache_dir=None, local_files_only=False):
    """Prefer author assets; otherwise recover the IDs exposed by reference samples.

    binary/{train,valid}_words.npy establishes title/template/stress IDs and count
    IDs 5..7 = 50271..50273; free_gen establishes the keywords ID. The linear
    extension to counts 1..40 is an assumption, not an author-supplied tokenizer.
    Other unused added IDs keep opaque names. The counter variant has a
    different count-to-template mapping and cannot verify this profile.
    """
    if directory is not None:
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=local_files_only)
        provenance = tokenizer.init_kwargs.get(
            "xai_tokenizer_provenance", {"source": "supplied", "path": str(directory)}
        )
    else:
        if cache_dir is None:
            local_cache = Path(__file__).resolve().parents[1] / "data/huggingface/hub"
            cache_dir = str(local_cache) if local_cache.exists() else None
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-base", cache_dir=cache_dir, local_files_only=local_files_only
        )
        if len(tokenizer) != BASE_VOCAB_SIZE:
            raise ValueError("Recovery requires the unmodified facebook/bart-base tokenizer")
        ids = {**REFERENCE_TOKEN_IDS, **{f"<syllable_{n}>": 50266 + n for n in range(1, 41)}}
        names = {value: key for key, value in ids.items()}
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    names.get(i, f"<xai_reserved_{i}>")
                    for i in range(BASE_VOCAB_SIZE, REFERENCE_VOCAB_SIZE)
                ]
            }
        )
        if any(tokenizer.convert_tokens_to_ids(k) != v for k, v in ids.items()):
            raise ValueError("Recovered tokenizer IDs do not match the reference samples")
        provenance = {
            "source": "reference_binary_recovery_v1",
            "reference_commit": "2fb45822c89b41e035455188644b640165dea463",
            "evidence": [
                "XAI-Lyricist/dataset/custom_sample/binary/{train,valid}_words.npy",
                "XAI-Lyricist/dataset/custom_sample/free_gen/{train,valid}_words.npy (keywords)",
            ],
            "verified_ids": {
                **REFERENCE_TOKEN_IDS,
                **{f"<syllable_{n}>": 50266 + n for n in (5, 6, 7)},
            },
            "assumptions": [
                "Unobserved count tokens extend the observed contiguous mapping: ID=50266+n.",
                "Base vocabulary and BPE merges are facebook/bart-base.",
                "Unused added IDs have opaque placeholder names; their semantics are unknown.",
                "The checkpoint used the same tokenizer as the bundled binary samples.",
            ],
            "conflicting_evidence": (
                "counter/{train,valid}_words.npy uses count prompts offset by +3 relative "
                "to template length. This profile follows binary, not counter."
            ),
        }
        logger.warning("Using recovered XAI token IDs; original tokenizer is unavailable")
    if len(tokenizer) != REFERENCE_VOCAB_SIZE:
        raise ValueError(f"The original checkpoint requires {REFERENCE_VOCAB_SIZE} tokenizer IDs")
    tokenizer.padding_side = "right"
    tokenizer.init_kwargs["xai_tokenizer_provenance"] = provenance
    return tokenizer


def validate_tokenizer(tokenizer, model):
    if len(tokenizer) != model.bart.config.vocab_size:
        raise ValueError("XAI tokenizer and checkpoint vocabulary sizes differ")
    if len(tokenizer) == REFERENCE_VOCAB_SIZE and any(
        tokenizer.convert_tokens_to_ids(token) != expected
        for token, expected in {
            **REFERENCE_TOKEN_IDS,
            **{f"<syllable_{n}>": 50266 + n for n in range(1, model.max_syllables + 1)},
        }.items()
    ):
        raise ValueError("XAI source token IDs differ from the checkpoint recovery profile")
    for token in (
        *REFERENCE_TOKEN_IDS,
        *(f"<syllable_{n}>" for n in range(1, model.max_syllables + 1)),
    ):
        token_id(tokenizer, token)
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(tokenizer, name) != getattr(model.bart.config, name):
            raise ValueError(f"XAI tokenizer has an incompatible {name}")


def encode_targets(record, tokenizer):
    # BOS is the decoder prefix, not a target. BART shifts these labels right with
    # decoder_start_token_id=BOS, exactly matching the reference inference prefix.
    labels = []
    for line in record_lines(record):
        words = line.get("words")
        if words:
            for word in words:
                labels.extend(tokenizer.encode(" " + word["text"], add_special_tokens=False))
        else:
            labels.extend(tokenizer.encode(" " + line["text"].strip(), add_special_tokens=False))
        labels.extend(tokenizer.encode(".", add_special_tokens=False))
    labels.append(tokenizer.eos_token_id)
    return {"labels": labels}


def encode_example(record, tokenizer, max_syllables):
    return {**encode_source(record, tokenizer, max_syllables), **encode_targets(record, tokenizer)}
