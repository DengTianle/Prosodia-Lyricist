"""Generate a song from a melody MIDI using a trained Prosodia checkpoint."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from . import ipa
from .evaluation import conditional_perplexity, evaluate_prosody, reference_targets, text_prosody
from .features import encode_source
from .generation import resolve_repetition_settings
from .midi import midi_note_comparison, midi_records
from .model import ProsodyBart
from .prepare import SCHEMA_VERSION
from .report import markdown_report, write_report
from .runtime import seed_everything, select_device
from .xai_model import XAILyricist, load_template


def package_versions():
    versions = {}
    for name in ("torch", "transformers", "prosodic", "miditoolkit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_stress_source(prosody_mode, stress_source, training_stress="ipa"):
    if prosody_mode not in ipa.PROSODY_MODES:
        raise ValueError("prosody_mode must be paper or actual")
    if prosody_mode == "actual":
        if stress_source is not None:
            raise ValueError("Repo mode selects MIDI and IPA rules together; omit --stress-source")
        return "actual"
    if stress_source == "actual":
        raise ValueError("Select --prosody-mode actual to use repo rules together")
    return stress_source or ("unknown" if training_stress == "unknown" else "supplement")


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
    prosody_mode=None,
    return_report=False,
    reference_lines=None,
    return_explanations=False,
    prosody_correction=True,
    bridge_checkpoint=None,
    template_mode="soft",
    no_repeat_ngram_size=None,
    repetition_penalty=None,
    tokenizer_dir=None,
    bart_config=None,
    local_files_only=False,
):
    if temperature <= 0 or top_k < 1 or (max_new_tokens is not None and max_new_tokens < 1):
        raise ValueError("temperature, top_k, and max_new_tokens must be positive")
    seed_everything(seed)
    checkpoint = Path(checkpoint)
    if prosody_mode not in (None, *ipa.PROSODY_MODES):
        raise ValueError("prosody_mode must be paper or actual")
    joint = (checkpoint / "end_to_end.json").exists()
    if template_mode not in ("soft", "hard") or (template_mode != "soft" and not joint):
        raise ValueError("Hard template ablation requires an end-to-end checkpoint")
    if joint and (bridge_checkpoint or stress_source is not None):
        raise ValueError("An end-to-end checkpoint already includes its melody conditioner")
    if bridge_checkpoint and stress_source is not None:
        raise ValueError("stress_source is a heuristic option; omit it when using a bridge")
    if bridge_checkpoint and (checkpoint / "melody.json").exists():
        raise ValueError("A bridge requires a four-stream template-decoder checkpoint")
    device = select_device(device)
    if checkpoint.is_file():
        model, tokenizer = load_template(
            checkpoint,
            tokenizer_dir=tokenizer_dir,
            bart_config=bart_config,
            local_files_only=local_files_only,
        )
        model = model.to(device).eval()
        run = dict(
            data_schema_version=SCHEMA_VERSION,
            provenance=model.provenance,
            config=dict(
                data=dict(stress_source="ipa"),
                model=dict(
                    max_source_length=model.bart.config.max_position_embeddings,
                    max_target_length=model.bart.config.max_position_embeddings,
                ),
            ),
        )
    else:
        if tokenizer_dir is not None or bart_config is not None:
            raise ValueError("Tokenizer/config overrides require an original XAI checkpoint file")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
        run = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
    if joint:
        from .end_to_end_model import EndToEndProsody

        joint_model = EndToEndProsody.load(checkpoint).to(device).eval()
        model = joint_model.template_model
    elif not checkpoint.is_file():
        model = ProsodyBart.load(checkpoint).to(device).eval()
    if run.get("data_schema_version") not in (3, 4, SCHEMA_VERSION):
        raise ValueError("Checkpoint uses an older input scheme; prepare and train song-level data")
    explainable = isinstance(model, ProsodyBart)
    xai = isinstance(model, XAILyricist)
    if xai:
        from .xai_features import validate_tokenizer

        validate_tokenizer(tokenizer, model)
    mode = prosody_mode or (joint_model.prosody_mode if joint else "actual" if xai else "paper")
    if joint and mode != joint_model.prosody_mode:
        raise ValueError("Requested IPA rules differ from the joint checkpoint's training rules")
    stress_source = resolve_stress_source(
        mode, stress_source, run["config"]["data"].get("stress_source", "ipa")
    )
    if bridge_checkpoint and not (explainable or xai):
        raise ValueError("A bridge requires an XAI or four-stream template checkpoint")
    if return_explanations and not (explainable or xai):
        raise ValueError("Legacy lyrics-only checkpoints have no generated explanation streams")
    target_limit = min(
        run["config"]["model"]["max_target_length"], model.bart.config.max_position_embeddings
    )
    if max_new_tokens is None:
        max_new_tokens = target_limit - 1  # Generation also includes the decoder start token.
    if max_new_tokens >= target_limit:
        raise ValueError("max_new_tokens must be smaller than the checkpoint's target token limit")
    bridge = None
    if joint:
        from .end_to_end_infer import condition_midi

        records, encoded, source_embeddings, bridge = condition_midi(
            joint_model,
            tokenizer,
            midi,
            title=title,
            track=track,
            device=device,
            max_source_length=run["config"]["model"]["max_source_length"],
            template_mode=template_mode,
        )
        bridge["checkpoint"] = str(checkpoint.resolve())
        stress_source = "learned"
    elif bridge_checkpoint:
        from .bridge_infer import predict_templates

        records, bridge = predict_templates(
            bridge_checkpoint,
            midi,
            title=title,
            track=track,
            device=device,
            max_syllables=model.max_syllables,
        )
        mode = prosody_mode or bridge.get("prosody_mode", "paper")
        if prosody_mode and mode != bridge.get("prosody_mode", "paper"):
            raise ValueError("Requested IPA rules differ from the learned bridge's rules")
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
    if xai:
        from .xai_features import encode_source as source_encoder
    elif not explainable:
        from .legacy_features import encode_source as source_encoder
    if not joint:
        encoded = source_encoder({"title": title, "lines": records}, tokenizer, model.max_syllables)
    source_limit = min(
        run["config"]["model"]["max_source_length"], model.bart.config.max_position_embeddings
    )
    if len(encoded["input_ids"]) > source_limit:
        raise ValueError("MIDI song template exceeds the checkpoint's source token limit")
    inputs = {key: torch.tensor([value], device=device) for key, value in encoded.items()}
    if joint:
        inputs["inputs_embeds"] = source_embeddings
    repetition = resolve_repetition_settings(model, no_repeat_ngram_size, repetition_penalty)
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": top_k > 1,
        **repetition,
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
    elif xai:
        result = model.generate(**inputs, tokenizer=tokenizer, **generation)
        tokens = result.sequences
        explanation = result.explanations[0]
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
    report = template_report(
        midi, records, title=title, track=track, stress_source=stress_source, prosody_mode=mode
    )
    if joint:
        report["template_role"] = bridge["template_role"]
        report["warnings"].append(
            "Template labels are an argmax preview of the soft source embeddings. "
            "Template-agreement scores are diagnostics, not reference-lyric accuracy."
            if template_mode == "soft"
            else "Hard-template ablation of an end-to-end checkpoint."
        )
    if bridge:
        report["bridge"] = bridge
        if bridge.get("unmarked_notes_omitted"):
            report["warnings"].append(
                f"Repo phrase parsing omits {bridge['unmarked_notes_omitted']} notes "
                "after the final phrase-end marker."
            )
    report.update(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_run=run,
        decoder_mode=(
            "explainable" if explainable else "xai_single_stream" if xai else "legacy_lyrics"
        ),
        generation={
            "seed": seed,
            "top_k": top_k,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "device": str(device),
            **repetition,
            "hit_token_limit": explanation["hit_token_limit"]
            if explainable or xai
            else len(tokens[0]) - 1 >= max_new_tokens,
        },
        encoded_source=encoded,
        generated_token_ids=tokens[0].tolist(),
        raw_text=text,
        lyrics=lyrics,
        metrics=evaluate_prosody(
            records, lyrics, **({"prosody_mode": "actual"} if mode == "actual" else {})
        ),
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
    elif xai:
        report["decoder_explanation"] = explanation
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
        if xai:
            from .xai_features import encode_targets

            streams = [
                encode_targets(
                    {
                        "lines": [
                            {
                                "words": ipa.parse_words(
                                    t, **({"prosody_mode": "actual"} if mode == "actual" else {})
                                )
                            }
                            for t in reference_lines
                        ]
                    },
                    tokenizer,
                )["labels"]
            ]
        else:
            streams = reference_targets(
                reference_lines, tokenizer, model.max_syllables, explainable=explainable
            )
        if len(streams[0]) > target_limit:
            raise ValueError("Reference exceeds checkpoint target token limit; no truncation")
        report["reference_lines"] = reference_lines
        report["reference_syllables"] = [
            text_prosody(line, **({"prosody_mode": "actual"} if mode == "actual" else {}))
            for line in reference_lines
        ]
        targets = [torch.tensor([stream], device=device) for stream in streams]
        report["reference_perplexity"] = conditional_perplexity(
            model,
            inputs,
            targets[0],
            tokenizer,
            prosody_labels=targets[1:] if explainable else None,
        )
    return report


def template_report(midi, records, *, title, track, stress_source, prosody_mode="paper"):
    learned = stress_source == "learned"
    report = {
        "report_version": 1,
        "unit": "whole_song",
        "title": title,
        "midi": str(Path(midi).resolve()),
        "midi_sha256": hashlib.sha256(Path(midi).read_bytes()).hexdigest(),
        "track": track,
        "stress_source": stress_source,
        "prosody_mode": prosody_mode,
        "template": records,
        "template_role": "model_input",
        "warnings": (
            [
                "Stress and vowel length are learned IPA-template predictions. "
                "Each MIDI note supplies one source prosody slot.",
                "No note-to-syllable alignment is inferred; note arrays are retained separately.",
            ]
            if learned
            else [
                "Repo getProsody(): duration-dependent floor stress grid; long iff duration "
                "exceeds its eight-quarter onset-bucket mean. File ticks-per-beat are used.",
            ]
            if prosody_mode == "actual"
            else [
                "One MIDI note is treated as one syllable; melisma is not inferred.",
                "The supplement's ambiguous onset/beat equation is resolved using its Figure 1 "
                "and explicit note-grid conventions documented in docs/evaluation.md.",
            ]
        )
        + (
            ["No time signature supplied: assumed 4/4."]
            if prosody_mode == "paper" and records[0].get("meter_assumed")
            else []
        )
        + (
            ["Notes after the final MIDI marker are retained as an additional trailing phrase."]
            if records[-1]["unmarked_tail"]
            else []
        ),
    }
    if prosody_mode == "actual":
        report["pronunciation"] = ipa.historical_provenance()
        omitted = records[-1].get("unmarked_notes_omitted", 0)
        if omitted:
            report["warnings"].append(
                f"Repo phrase parsing omits {omitted} notes after the final phrase-end marker."
            )
    if learned:
        report["note_comparison"] = midi_note_comparison(
            midi,
            title=title,
            track=track,
            **({"prosody_mode": "actual"} if prosody_mode == "actual" else {}),
        )
        if report["note_comparison"]["warning"]:
            report["warnings"].append(report["note_comparison"]["warning"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="The run's best/ directory")
    parser.add_argument("--bridge-checkpoint", help="Learned melody-to-template best/ directory")
    parser.add_argument(
        "--template-mode",
        choices=("soft", "hard"),
        default="soft",
        help="End-to-end checkpoints: soft conditioning or hard argmax ablation",
    )
    parser.add_argument("--midi", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--track", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        help="Lyric BPE n-gram filter; defaults to the checkpoint setting, 0 disables it",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Text repetition penalty; defaults to the checkpoint setting, 1 disables it",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, help="Defaults to the checkpoint's target limit minus one"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stress-source", choices=("supplement", "heuristic", "unknown"))
    parser.add_argument(
        "--prosody-mode",
        choices=ipa.PROSODY_MODES,
        help="Defaults to repo rules for XAI and the saved rules for joint models",
    )
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--bart-config")
    parser.add_argument("--local-files-only", action="store_true")
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
        "--explanations", help="New decoder details JSON; four-stream models include prosody"
    )
    parser.add_argument(
        "--no-prosody-correction",
        action="store_false",
        dest="prosody_correction",
        help="Ablation: feed predicted prosody back without IPA correction",
    )
    args = parser.parse_args()
    if args.template_mode == "hard" and not (
        args.checkpoint and (Path(args.checkpoint) / "end_to_end.json").exists()
    ):
        parser.error("--template-mode hard requires an end-to-end checkpoint")
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
        mode = args.prosody_mode or "actual"
        if args.checkpoint and (Path(args.checkpoint) / "end_to_end.json").exists():
            from .end_to_end_infer import condition_midi
            from .end_to_end_model import EndToEndProsody

            if args.bridge_checkpoint or args.stress_source:
                parser.error("An end-to-end checkpoint already includes its melody conditioner")
            seed_everything(args.seed)
            device = select_device(args.device)
            model = EndToEndProsody.load(args.checkpoint).to(device).eval()
            mode = args.prosody_mode or model.prosody_mode
            if mode != model.prosody_mode:
                parser.error("Requested IPA rules differ from the joint checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(
                Path(args.checkpoint) / "tokenizer", local_files_only=True
            )
            run = json.loads((Path(args.checkpoint) / "run.json").read_text())
            records, _, _, bridge = condition_midi(
                model,
                tokenizer,
                args.midi,
                title=args.title,
                track=args.track,
                device=device,
                max_source_length=run["config"]["model"]["max_source_length"],
                template_mode=args.template_mode,
            )
            bridge["checkpoint"] = str(Path(args.checkpoint).resolve())
            stress = "learned"
        elif args.bridge_checkpoint:
            from .bridge_infer import predict_templates

            seed_everything(args.seed)
            records, bridge = predict_templates(
                args.bridge_checkpoint,
                args.midi,
                title=args.title,
                track=args.track,
                device=select_device(args.device),
            )
            mode = args.prosody_mode or bridge.get("prosody_mode", "paper")
            if args.prosody_mode and mode != bridge.get("prosody_mode", "paper"):
                parser.error("Requested IPA rules differ from the learned bridge's rules")
            stress = "learned"
        else:
            stress = resolve_stress_source(mode, args.stress_source)
            records = midi_records(
                args.midi, title=args.title, track=args.track, stress_source=stress
            )
        report = template_report(
            args.midi,
            records,
            title=args.title,
            track=args.track,
            stress_source=stress,
            prosody_mode=mode,
        )
        if bridge:
            report["bridge"] = bridge
            if "template_role" in bridge:
                report["template_role"] = bridge["template_role"]
                report["warnings"].append(
                    "Displayed hard labels are a preview; see soft slot probabilities in JSON."
                )
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
