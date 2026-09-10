"""Shared template encoding for DALI training and MIDI inference."""

STRESSES = ("strong", "substrong", "weak", "unknown")
LENGTH_IDS = {"long": 1, "short": 2}
SOURCE_KEYS = ("input_ids", "length_ids", "remainder_ids", "attention_mask")


def special_tokens(max_syllables):
    return (
        ["<title>", "<template>", "<line>"]
        + [f"<{stress}>" for stress in STRESSES]
        + [f"<syllable_{n}>" for n in range(1, max_syllables + 1)]
    )


def configure_tokenizer(tokenizer, max_syllables):
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens(max_syllables)})
    tokenizer.padding_side = "right"
    return tokenizer


def encode_source(record, tokenizer, max_syllables):
    syllables = record["syllables"]
    count = len(syllables)
    if not 1 <= count <= max_syllables:
        raise ValueError(f"Expected 1..{max_syllables} syllables, got {count}")

    def token_id(token):
        value = tokenizer.convert_tokens_to_ids(token)
        if value is None or value == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer is missing the template token {token}")
        return value

    ids = [tokenizer.bos_token_id, token_id("<title>")]
    ids += tokenizer.encode(record.get("title", ""), add_special_tokens=False)
    ids += [token_id(f"<syllable_{count}>"), token_id("<template>")]
    lengths, remainders = [0] * len(ids), [0] * len(ids)
    for index, syllable in enumerate(syllables):
        ids.append(token_id(f"<{syllable['stress']}>"))
        lengths.append(LENGTH_IDS[syllable["length"]])
        # Zero is padding; remaining=0 must still have a non-padding embedding.
        remainders.append(count - index)
    ids += [token_id("<line>"), tokenizer.eos_token_id]
    lengths += [0, 0]
    remainders += [0, 0]
    return {
        "input_ids": ids,
        "length_ids": lengths,
        "remainder_ids": remainders,
        "attention_mask": [1] * len(ids),
    }


def encode_example(record, tokenizer, max_syllables, *, scaffold=False):
    example = encode_source(record, tokenizer, max_syllables)
    if scaffold or record.get("words"):
        words = record.get("words")
        if not words:
            raise ValueError("Auxiliary losses require prepared word scaffolding; prepare again")
        remaining = sum(w["syllable_count"] for w in words)
        labels = [tokenizer.bos_token_id]
        syllable, remainder, sentence = [-100], [-100], [0]
        for word in words:
            count = word["syllable_count"]
            remaining -= count
            tokens = tokenizer.encode(" " + word["text"], add_special_tokens=False)
            labels.extend(tokens)
            syllable.extend([count] * len(tokens))
            remainder.extend([remaining] * len(tokens))
            sentence.extend([1] * len(tokens))
        labels.append(tokenizer.eos_token_id)
        example.update(
            labels=labels,
            syllable_labels=syllable + [-100],
            remainder_labels=remainder + [-100],
            sentence_labels=sentence + [0],
        )
        if not scaffold:
            for key in ("syllable_labels", "remainder_labels", "sentence_labels"):
                del example[key]
        return example
    # BART shifts labels internally, starting with decoder_start_token_id.
    example["labels"] = tokenizer.encode(record["text"], add_special_tokens=True)
    return example
