"""Shared song-level template encoding for DALI training and MIDI inference."""

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


def encode_source(record, tokenizer, max_syllables):
    def token_id(token):
        value = tokenizer.convert_tokens_to_ids(token)
        if value is None or value == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer is missing the template token {token}")
        return value

    ids = [tokenizer.bos_token_id, token_id("<title>")]
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
        append_plain([token_id(f"<syllable_{count}>"), token_id("<template>")])
        for index, syllable in enumerate(syllables):
            ids.append(token_id(f"<{syllable['stress']}>"))
            lengths.append(LENGTH_IDS[syllable["length"]])
            # Remaining+1 reserves zero for padding; reset for each line.
            remainders.append(count - index)
        # Original per-line prompt and period boundary. DALI has no keyword field,
        # so its prompt is empty; no target words are copied into conditioning.
        append_plain([token_id("<keywords>")])
        append_plain(
            tokenizer.encode(line.get("keyword", "").replace(".", ""), add_special_tokens=False)
        )
        append_plain(tokenizer.encode(".", add_special_tokens=False))
    append_plain([tokenizer.eos_token_id])
    return {
        "input_ids": ids,
        "length_ids": lengths,
        "remainder_ids": remainders,
        "attention_mask": [1] * len(ids),
    }


def encode_example(record, tokenizer, max_syllables, *, scaffold=False):
    example = encode_source(record, tokenizer, max_syllables)
    labels = [tokenizer.bos_token_id]
    syllable, remainder, sentence = [-100], [-100], [0]
    for line in record_lines(record):
        words = line.get("words")
        if scaffold and not words:
            raise ValueError("Auxiliary losses require prepared word scaffolding; prepare again")
        if words:
            remaining = sum(w["syllable_count"] for w in words)
            if remaining != len(line["syllables"]):
                raise ValueError("Word syllable counts do not match the line template")
            for word in words:
                count = word["syllable_count"]
                remaining -= count
                tokens = tokenizer.encode(" " + word["text"], add_special_tokens=False)
                labels.extend(tokens)
                syllable.extend([count] * len(tokens))
                remainder.extend([remaining] * len(tokens))
                sentence.extend([1] * len(tokens))
        else:
            tokens = tokenizer.encode(line["text"], add_special_tokens=False)
            labels.extend(tokens)
            syllable.extend([-100] * len(tokens))
            remainder.extend([-100] * len(tokens))
            sentence.extend([1] * len(tokens))
        tokens = tokenizer.encode(".", add_special_tokens=False)
        labels.extend(tokens)
        syllable.extend([-100] * len(tokens))
        remainder.extend([-100] * len(tokens))
        sentence.extend([0] * len(tokens))
    labels.append(tokenizer.eos_token_id)
    example["labels"] = labels
    if scaffold:
        example.update(
            syllable_labels=syllable + [-100],
            remainder_labels=remainder + [-100],
            sentence_labels=sentence + [0],
        )
    return example
