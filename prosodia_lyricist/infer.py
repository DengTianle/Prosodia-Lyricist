"""Generate a song from a melody MIDI using a trained Prosodia checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .features import encode_source
from .midi import midi_records
from .model import ProsodyBart
from .prepare import SCHEMA_VERSION
from .runtime import seed_everything, select_device


def infer(
    checkpoint,
    midi,
    *,
    title="",
    track=0,
    device="auto",
    temperature=1.0,
    top_k=3,
    max_new_tokens=None,
    seed=1234,
    stress_source=None,
):
    if temperature <= 0 or top_k < 1 or (max_new_tokens is not None and max_new_tokens < 1):
        raise ValueError("temperature, top_k, and max_new_tokens must be positive")
    seed_everything(seed)
    checkpoint = Path(checkpoint)
    run = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
    if run.get("data_schema_version") != SCHEMA_VERSION:
        raise ValueError("Checkpoint uses an older input scheme; prepare and train song-level data")
    # Match duration/count-only training by default when lexical stress was disabled.
    if stress_source is None:
        stress_source = (
            "unknown" if run["config"]["data"]["stress_source"] == "unknown" else "heuristic"
        )
    device = select_device(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    model = ProsodyBart.load(checkpoint).to(device).eval()
    target_limit = min(
        run["config"]["model"]["max_target_length"], model.bart.config.max_position_embeddings
    )
    if max_new_tokens is None:
        max_new_tokens = target_limit - 1  # Generation also includes the decoder start token.
    if max_new_tokens >= target_limit:
        raise ValueError("max_new_tokens must be smaller than the checkpoint's target token limit")
    records = midi_records(
        midi,
        title=title,
        track=track,
        max_syllables=model.max_syllables,
        stress_source=stress_source,
    )
    encoded = encode_source({"title": title, "lines": records}, tokenizer, model.max_syllables)
    source_limit = min(
        run["config"]["model"]["max_source_length"], model.bart.config.max_position_embeddings
    )
    if len(encoded["input_ids"]) > source_limit:
        raise ValueError("MIDI song template exceeds the checkpoint's source token limit")
    inputs = {key: torch.tensor([value], device=device) for key, value in encoded.items()}
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": top_k > 1,
        "bad_words_ids": [
            [tokenizer.convert_tokens_to_ids(token)]
            for token in tokenizer.additional_special_tokens
        ],
    }
    if top_k > 1:
        generation.update(temperature=temperature, top_k=top_k)
    tokens = model.generate(**inputs, **generation)
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return [line.strip() for line in text.split(".") if line.strip()] or [""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="The run's best/ directory")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--track", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-new-tokens", type=int, help="Defaults to the checkpoint's target limit minus one"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stress-source", choices=("heuristic", "unknown"))
    parser.add_argument("--output", help="Optional new UTF-8 text file")
    args = parser.parse_args()
    output = args.output
    del args.output
    text = "\n".join(infer(**vars(args))) + "\n"
    print(text, end="")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
