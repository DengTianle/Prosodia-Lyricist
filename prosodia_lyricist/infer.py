"""Generate a song from a melody MIDI using a trained Prosodia checkpoint."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .evaluation import conditional_perplexity, evaluate_prosody
from .features import encode_source
from .midi import midi_records
from .model import ProsodyBart
from .prepare import SCHEMA_VERSION
from .report import markdown_report, write_report
from .runtime import seed_everything, select_device


def package_versions():
    versions = {}
    for name in ("torch", "transformers", "prosodic", "miditoolkit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@torch.inference_mode()
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
    return_report=False,
    reference_lines=None,
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
            "unknown" if run["config"]["data"]["stress_source"] == "unknown" else "supplement"
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
    if reference_lines is not None and (
        len(reference_lines) != len(records) or any(not line.strip() for line in reference_lines)
    ):
        raise ValueError("Reference must have one nonempty line per MIDI phrase")
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
    lyrics = [line.strip() for line in text.split(".") if line.strip()]
    if not return_report:
        return lyrics or [""]
    report = template_report(midi, records, title=title, track=track, stress_source=stress_source)
    report.update(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_run=run,
        generation={
            "seed": seed,
            "top_k": top_k,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "device": str(device),
            "hit_token_limit": len(tokens[0]) - 1 >= max_new_tokens,
        },
        encoded_source=encoded,
        generated_token_ids=tokens[0].tolist(),
        raw_text=text,
        lyrics=lyrics,
        metrics=evaluate_prosody(records, lyrics),
        generated_perplexity=conditional_perplexity(model, inputs, tokens[:, 1:], tokenizer),
        software_versions=package_versions(),
    )
    if report["generation"]["hit_token_limit"]:
        report["warnings"].append(
            "Generation reached its token budget; the final EOS may have been forced. "
            "The output may be incomplete."
        )
    if len(lyrics) != len(records):
        report["warnings"].append(
            "Generated phrase count differs from the MIDI. Phrases are paired by ordinal "
            "position; later pairs may be misaligned."
        )
    if reference_lines is not None:
        # Match training's wordwise byte-BPE construction, including punctuation.
        from .ipa import parse_words

        labels = [tokenizer.bos_token_id]
        for line in reference_lines:
            for word in parse_words(line):
                labels.extend(tokenizer.encode(" " + word["text"], add_special_tokens=False))
            labels.extend(tokenizer.encode(".", add_special_tokens=False))
        labels.append(tokenizer.eos_token_id)
        if len(labels) > target_limit:
            raise ValueError("Reference exceeds checkpoint target token limit; no truncation")
        report["reference_lines"] = reference_lines
        report["reference_perplexity"] = conditional_perplexity(
            model, inputs, torch.tensor([labels], device=device), tokenizer
        )
    return report


def template_report(midi, records, *, title, track, stress_source):
    return {
        "report_version": 1,
        "unit": "whole_song",
        "title": title,
        "midi": str(Path(midi).resolve()),
        "midi_sha256": hashlib.sha256(Path(midi).read_bytes()).hexdigest(),
        "track": track,
        "stress_source": stress_source,
        "template": records,
        "warnings": [
            "One MIDI note is treated as one syllable; melisma is not inferred.",
            "The supplement's ambiguous onset/beat equation is resolved using its Figure 1 "
            "and explicit note-grid conventions documented in docs/evaluation.md.",
        ]
        + (["No time signature supplied: assumed 4/4."] if records[0]["meter_assumed"] else [])
        + (
            ["Notes after the final MIDI marker are retained as an additional trailing phrase."]
            if records[-1]["unmarked_tail"]
            else []
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="The run's best/ directory")
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
    parser.add_argument("--stress-source", choices=("supplement", "heuristic", "unknown"))
    parser.add_argument("--output", help="Optional new UTF-8 text file")
    parser.add_argument("--report-prefix", help="Write PREFIX.md and PREFIX.json with metrics")
    parser.add_argument("--reference", help="UTF-8 lyrics, one line per phrase, for reference PPL")
    parser.add_argument("--template-only", action="store_true", help="Inspect MIDI without a model")
    args = parser.parse_args()
    if not args.template_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --template-only is used")
    if args.template_only and args.reference:
        parser.error("--reference requires model inference")
    if args.reference and not args.report_prefix:
        parser.error("--reference requires --report-prefix to save its score")
    output = args.output
    paths = ([Path(output)] if output else []) + (
        [Path(args.report_prefix + suffix) for suffix in (".md", ".json")]
        if args.report_prefix
        else []
    )
    if len({p.resolve() for p in paths}) != len(paths) or any(p.exists() for p in paths):
        parser.error("Output paths must be distinct new files")
    if args.template_only:
        stress = args.stress_source or "supplement"
        records = midi_records(args.midi, title=args.title, track=args.track, stress_source=stress)
        report = template_report(
            args.midi, records, title=args.title, track=args.track, stress_source=stress
        )
        text = markdown_report(report)
    else:
        kwargs = vars(args).copy()
        for key in ("output", "report_prefix", "reference", "template_only"):
            kwargs.pop(key)
        result = infer(
            **kwargs,
            return_report=bool(args.report_prefix),
            reference_lines=Path(args.reference).read_text(encoding="utf-8").splitlines()
            if args.reference
            else None,
        )
        report = result if args.report_prefix else None
        text = "\n".join(result["lyrics"] if report else result) + "\n"
    print(text, end="")
    if args.report_prefix:
        write_report(report, args.report_prefix)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
