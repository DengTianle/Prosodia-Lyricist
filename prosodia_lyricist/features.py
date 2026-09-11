"""Paper templates and four aligned decoder streams, with explicit BPE word boundaries."""

from .legacy_features import record_lines

STRESSES = ("strong", "weak")
STRESS_IDS = {"strong": 1, "weak": 2}
LENGTH_IDS = {"long": 1, "short": 2}
SOURCE_KEYS = ("input_ids", "length_ids", "attention_mask")
TARGET_KEYS = ("labels", "syllable_labels", "stress_labels", "length_labels")
MAX_LINES = 256
WORD_END = "<word_end>"


def special_tokens(max_syllables):
    # Keep the previous vocabulary available for explicitly selected legacy experiments.
    from .legacy_features import special_tokens as legacy_tokens

    return (
        legacy_tokens(max_syllables)
        + ["<prosody>", WORD_END]
        + [f"<sent_{i}>" for i in range(MAX_LINES)]
    )


def configure_tokenizer(tokenizer, max_syllables):
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens(max_syllables)})
    tokenizer.padding_side = "right"
    return tokenizer


def token_id(tokenizer, token):
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or value == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer is missing the template token {token}")
    return value


def encode_source(record, tokenizer, max_syllables):
    ids = [tokenizer.bos_token_id, token_id(tokenizer, "<title>")]
    ids += tokenizer.encode(record.get("title", ""), add_special_tokens=False)
    lengths = [0] * len(ids)

    def plain(tokens):
        ids.extend(tokens)
        lengths.extend([0] * len(tokens))

    lines = record_lines(record)
    if len(lines) > MAX_LINES:
        raise ValueError(f"At most {MAX_LINES} lines can be represented")
    for index, line in enumerate(lines):
        syllables = line["syllables"]
        if not 1 <= len(syllables) <= max_syllables:
            raise ValueError(f"Expected 1..{max_syllables} syllables per line")
        plain([token_id(tokenizer, f"<sent_{index}>"), token_id(tokenizer, "<keywords>")])
        plain(tokenizer.encode(line.get("keyword", ""), add_special_tokens=False))
        plain([token_id(tokenizer, "<prosody>")])
        for syllable in syllables:
            stress = syllable["stress"]
            # MIDI sub-strong beats and secondary lexical stress are strong in Table 1.
            if stress == "substrong":
                stress = "strong"
            if stress not in (*STRESSES, "unknown"):
                raise ValueError(f"Invalid stress: {stress}")
            ids.append(token_id(tokenizer, f"<{stress}>"))
            lengths.append(LENGTH_IDS[syllable["length"]])
    plain([tokenizer.eos_token_id])
    return {"input_ids": ids, "length_ids": lengths, "attention_mask": [1] * len(ids)}


def word_prosody(syllables):
    """Section 3.1: whether a word contains stressed / long syllables."""
    if not syllables:
        return (0, 0, 0)
    if any(s["stress"] not in ("strong", "substrong", "weak") for s in syllables):
        raise ValueError("Explainable targets require IPA stress; prepare with stress_source: ipa")
    return (
        len(syllables),
        1 if any(s["stress"] in ("strong", "substrong") for s in syllables) else 2,
        1 if any(s["length"] == "long" for s in syllables) else 2,
    )


def encode_example(record, tokenizer, max_syllables):
    example = encode_source(record, tokenizer, max_syllables)
    streams = [[tokenizer.bos_token_id], [0], [0], [0]]

    def append(token, prosody=(0, 0, 0)):
        for stream, value in zip(streams, (token, *prosody), strict=True):
            stream.append(value)

    for line in record_lines(record):
        words = line.get("words")
        if not words:
            raise ValueError("Explainable targets require prepared IPA words; prepare again")
        offset = 0
        for word in words:
            count = word["syllable_count"]
            if not isinstance(count, int) or not 0 <= count <= max_syllables:
                raise ValueError("Invalid word syllable count")
            syllables = line["syllables"][offset : offset + count]
            if len(syllables) != count:
                raise ValueError("Word syllable counts do not match the line template")
            offset += count
            tokens = tokenizer.encode(" " + word["text"], add_special_tokens=False)
            if not tokens or tokenizer.unk_token_id in tokens:
                raise ValueError(f"Word cannot be represented by the tokenizer: {word['text']!r}")
            for token in tokens:
                append(token)
            if count:
                # BART BPE pieces precede one completed-word event. Predict its three
                # labels once, and feed that event back before generating the next word.
                append(token_id(tokenizer, WORD_END), word_prosody(syllables))
        if offset != len(line["syllables"]):
            raise ValueError("Word syllable counts do not match the line template")
        for token in tokenizer.encode(".", add_special_tokens=False):
            append(token)
    append(tokenizer.eos_token_id)
    example.update(zip(TARGET_KEYS, streams, strict=True))
    return example
