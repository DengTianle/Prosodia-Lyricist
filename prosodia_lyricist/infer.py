"""Generate a song from a melody MIDI using a trained Prosodia checkpoint."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .evaluation import conditional_perplexity, evaluate_prosody, reference_targets, text_prosody
from .features import encode_source
from .midi import midi_note_comparison, midi_records
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
    return_explanations=False,
    prosody_correction=True,
    bridge_checkpoint=None,
    bridge_skeleton=None,
):
    if temperature <= 0 or top_k < 1 or (max_new_tokens is not None and max_new_tokens < 1):
        raise ValueError("temperature, top_k, and max_new_tokens must be positive")
    seed_everything(seed)
    checkpoint = Path(checkpoint)
    if bridge_skeleton is not None and not bridge_checkpoint:
        raise ValueError("bridge_skeleton requires a bridge_checkpoint")
    if bridge_checkpoint and stress_source is not None:
        raise ValueError("stress_source is a heuristic option; omit it when using a bridge")
    if bridge_checkpoint and (checkpoint / "melody.json").exists():
        raise ValueError("A bridge requires a four-stream template-decoder checkpoint")
    run = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
    if run.get("data_schema_version") not in (3, SCHEMA_VERSION):
        raise ValueError("Checkpoint uses an older input scheme; prepare and train song-level data")
    # Match duration/count-only training by default when lexical stress was disabled.
    if stress_source is None:
        stress_source = (
            "unknown" if run["config"]["data"]["stress_source"] == "unknown" else "supplement"
        )
    device = select_device(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    model = ProsodyBart.load(checkpoint).to(device).eval()
    explainable = isinstance(model, ProsodyBart)
    if bridge_checkpoint and not explainable:
        raise ValueError("A bridge requires a four-stream template-decoder checkpoint")
    if return_explanations and not explainable:
        raise ValueError("Legacy lyrics-only checkpoints have no generated explanation streams")
    target_limit = min(
        run["config"]["model"]["max_target_length"], model.bart.config.max_position_embeddings
    )
    if max_new_tokens is None:
        max_new_tokens = target_limit - 1  # Generation also includes the decoder start token.
    if max_new_tokens >= target_limit:
        raise ValueError("max_new_tokens must be smaller than the checkpoint's target token limit")
    bridge = None
    if bridge_checkpoint:
        from .bridge_infer import predict_templates

        records, bridge = predict_templates(
            bridge_checkpoint,
            midi,
            title=title,
            track=track,
            device=device,
            max_syllables=model.max_syllables,
            skeleton=bridge_skeleton,
        )
        stress_source = "learned"
    else:
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
    source_encoder = encode_source
    if not explainable:
        from .legacy_features import encode_source as source_encoder
    encoded = source_encoder({"title": title, "lines": records}, tokenizer, model.max_syllables)
    source_limit = min(
        run["config"]["model"]["max_source_length"], model.bart.config.max_position_embeddings
    )
    if len(encoded["input_ids"]) > source_limit:
        raise ValueError("MIDI song template exceeds the checkpoint's source token limit")
    inputs = {key: torch.tensor([value], device=device) for key, value in encoded.items()}
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": top_k > 1,
    }
    if top_k > 1:
        generation.update(temperature=temperature, top_k=top_k)
    explanation, prosody_labels = None, None
    if explainable:
        result = model.generate(
            **inputs, tokenizer=tokenizer, prosody_correction=prosody_correction, **generation
        )
        tokens = result.sequences
        prosody_labels = [
            stream[:, 1:] for stream in (result.syllable_ids, result.stress_ids, result.length_ids)
        ]
        explanation = result.explanations[0]
        explanation["template"] = {
            "lyrics": tokens[0].tolist(),
            "syllables": result.syllable_ids[0].tolist(),
            "stresses": result.stress_ids[0].tolist(),
            "lengths": result.length_ids[0].tolist(),
            "tokens": tokenizer.convert_ids_to_tokens(tokens[0].tolist()),
        }
        explanation["source"] = {"title": title, "lines": records}
        if bridge:
            explanation["bridge"] = bridge
        text = explanation["text"]
    else:
        generation["bad_words_ids"] = [
            [tokenizer.convert_tokens_to_ids(token)]
            for token in tokenizer.additional_special_tokens
        ]
        tokens = model.generate(**inputs, **generation)
        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    lyrics = [line.strip() for line in text.split(".") if line.strip()]
    if not return_report:
        if return_explanations:
            return explanation
        return lyrics or [""]
    report = template_report(midi, records, title=title, track=track, stress_source=stress_source)
    if bridge:
        report["bridge"] = bridge
    report.update(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_run=run,
        decoder_mode="explainable" if explainable else "legacy_lyrics",
        generation={
            "seed": seed,
            "top_k": top_k,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "device": str(device),
            "hit_token_limit": explanation["hit_token_limit"]
            if explainable
            else len(tokens[0]) - 1 >= max_new_tokens,
        },
        encoded_source=encoded,
        generated_token_ids=tokens[0].tolist(),
        raw_text=text,
        lyrics=lyrics,
        metrics=evaluate_prosody(records, lyrics),
        generated_perplexity=conditional_perplexity(
            model, inputs, tokens[:, 1:], tokenizer, prosody_labels=prosody_labels
        ),
        software_versions=package_versions(),
    )
    if explainable:
        report["decoder_explanation"] = explanation
        report["warnings"].append(
            "Lyric-BPE perplexity conditions on prior word-boundary/prosody events; "
            "it is not directly equivalent to baseline text-only perplexity."
        )
        if explanation["truncated_word"]:
            report["warnings"].append(
                "An unfinished final word was discarded; see decoder details."
            )
    if report["generation"]["hit_token_limit"]:
        report["warnings"].append(
            "Generation reached its event budget without completing the song."
            if explainable
            else "Generation reached its token budget; the final EOS may have been forced. "
            "The output may be incomplete."
        )
    if len(lyrics) != len(records):
        report["warnings"].append(
            "Generated phrase count differs from the MIDI. Phrases are paired by ordinal "
            "position; later pairs may be misaligned."
        )
    if reference_lines is not None:
        streams = reference_targets(
            reference_lines, tokenizer, model.max_syllables, explainable=explainable
        )
        if len(streams[0]) > target_limit:
            raise ValueError("Reference exceeds checkpoint target token limit; no truncation")
        report["reference_lines"] = reference_lines
        report["reference_syllables"] = [text_prosody(line) for line in reference_lines]
        targets = [torch.tensor([stream], device=device) for stream in streams]
        report["reference_perplexity"] = conditional_perplexity(
            model,
            inputs,
            targets[0],
            tokenizer,
            prosody_labels=targets[1:] if explainable else None,
        )
    return report


def template_report(midi, records, *, title, track, stress_source):
    learned = stress_source == "learned"
    report = {
        "report_version": 1,
        "unit": "whole_song",
        "title": title,
        "midi": str(Path(midi).resolve()),
        "midi_sha256": hashlib.sha256(Path(midi).read_bytes()).hexdigest(),
        "track": track,
        "stress_source": stress_source,
        "template": records,
        "template_role": "model_input",
        "warnings": (
            [
                "Stress, vowel length and syllable count are learned IPA-template predictions.",
                "No note-to-syllable alignment is inferred; note arrays are retained separately.",
            ]
            if learned
            else [
                "One MIDI note is treated as one syllable; melisma is not inferred.",
                "The supplement's ambiguous onset/beat equation is resolved using its Figure 1 "
                "and explicit note-grid conventions documented in docs/evaluation.md.",
            ]
        )
        + (["No time signature supplied: assumed 4/4."] if records[0].get("meter_assumed") else [])
        + (
            ["Notes after the final MIDI marker are retained as an additional trailing phrase."]
            if records[-1]["unmarked_tail"]
            else []
        ),
    }
    if learned:
        report["note_comparison"] = midi_note_comparison(midi, title=title, track=track)
        if report["note_comparison"]["warning"]:
            report["warnings"].append(report["note_comparison"]["warning"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="The run's best/ directory")
    parser.add_argument("--bridge-checkpoint", help="Learned melody-to-template best/ directory")
    parser.add_argument(
        "--bridge-skeleton",
        metavar="JSON",
        help="Ordered line_id entries with optional syllable_count; omitted counts are predicted",
    )
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
    parser.add_argument(
        "--reference",
        metavar="FILE",
        help="UTF-8 actual lyrics, one line per phrase, for ground-truth labels and reference PPL",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Inspect heuristic or learned templates without generating lyrics",
    )
    parser.add_argument(
        "--explanations", help="New JSON file with predicted/corrected decoder prosody"
    )
    parser.add_argument(
        "--no-prosody-correction",
        action="store_false",
        dest="prosody_correction",
        help="Ablation: feed predicted prosody back without IPA correction",
    )
    args = parser.parse_args()
    if args.bridge_skeleton and not args.bridge_checkpoint:
        parser.error("--bridge-skeleton requires --bridge-checkpoint")
    if args.bridge_checkpoint and args.stress_source:
        parser.error("--stress-source is a heuristic option; omit it with --bridge-checkpoint")
    if not args.template_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --template-only is used")
    if args.template_only and args.reference:
        parser.error("--reference requires model inference")
    if args.template_only and args.explanations:
        parser.error("--explanations requires model inference")
    if args.reference and not args.report_prefix:
        parser.error("--reference requires --report-prefix to save its score")
    output = args.output
    paths = ([Path(output)] if output else []) + (
        [Path(args.report_prefix + suffix) for suffix in (".md", ".json")]
        if args.report_prefix
        else []
    )
    if args.explanations:
        paths.append(Path(args.explanations))
    if len({p.resolve() for p in paths}) != len(paths) or any(p.exists() for p in paths):
        parser.error("Output paths must be distinct new files")
    if args.template_only:
        bridge = None
        if args.bridge_checkpoint:
            from .bridge_infer import predict_templates

            seed_everything(args.seed)
            records, bridge = predict_templates(
                args.bridge_checkpoint,
                args.midi,
                title=args.title,
                track=args.track,
                device=select_device(args.device),
                skeleton=args.bridge_skeleton,
            )
            stress = "learned"
        else:
            stress = args.stress_source or "supplement"
            records = midi_records(
                args.midi, title=args.title, track=args.track, stress_source=stress
            )
        report = template_report(
            args.midi, records, title=args.title, track=args.track, stress_source=stress
        )
        if bridge:
            report["bridge"] = bridge
        text = markdown_report(report)
    else:
        kwargs = vars(args).copy()
        for key in ("output", "report_prefix", "reference", "template_only", "explanations"):
            kwargs.pop(key)
        result = infer(
            **kwargs,
            return_report=bool(args.report_prefix),
            return_explanations=bool(args.explanations),
            reference_lines=Path(args.reference).read_text(encoding="utf-8").splitlines()
            if args.reference
            else None,
        )
        report = result if args.report_prefix else None
        lyrics = result["lyrics"] if report else result["lines"] if args.explanations else result
        text = "\n".join(lyrics) + "\n"
        if args.explanations:
            path = Path(args.explanations)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                json.dump(
                    result["decoder_explanation"] if report else result,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
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
