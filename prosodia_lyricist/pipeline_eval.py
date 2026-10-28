"""Evaluate MIDI -> template -> BART lyrics on paired MIDI/lyric songs."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from tqdm import tqdm

from .evaluation import (
    ACTUAL_METRIC_VERSION,
    METRIC_VERSION,
    aggregate_perplexity,
    prosody_bleu,
)
from .infer import TemplateDecoder, package_versions
from .ipa import PAPER_RULES, REPO_RULES
from .melody_checkpoint import file_sha256
from .midi_eval import (
    MIDI_METHODS,
    BridgePredictor,
    MidiPredictor,
    aggregate_scores,
    discover_songs,
    read_references,
    score_templates,
)
from .pipeline_reports import export_readable_reports

RELATIONS = ("template_reference", "lyrics_template", "lyrics_reference")


def compare_prosody(predictions, references):
    """Keep the MIDI evaluator's positional accuracy and add phrase-mean BLEU."""
    result = score_templates(predictions, [{**r, "text": r.get("text", "")} for r in references])
    for phrase, (prediction, reference) in zip(
        result["phrases"], zip_longest(predictions, references), strict=True
    ):
        phrase["prosody_bleu"] = prosody_bleu(
            reference["syllables"] if reference is not None else [],
            prediction["syllables"] if prediction is not None else [],
        )
    result["prosody_bleu_sum"] = sum(p["prosody_bleu"] for p in result["phrases"])
    result["prosody_bleu"] = (
        result["prosody_bleu_sum"] / len(result["phrases"]) if result["phrases"] else None
    )
    return result


def aggregate_relation(scores):
    result = aggregate_scores(scores)
    phrase_count = sum(len(score["phrases"]) for score in scores)
    result["prosody_bleu"] = (
        sum(score["prosody_bleu_sum"] for score in scores) / phrase_count if phrase_count else None
    )
    return result


def summarize_methods(results, predictors):
    common = [r for r in results if all("error" not in r["methods"][n] for n in predictors)]
    summaries = {}
    for name, predictor in predictors.items():
        successful = [r["methods"][name] for r in results if "error" not in r["methods"][name]]

        def summarize(methods):
            summary = {
                "evaluated_songs": len(methods),
                **{key: aggregate_relation([m[key] for m in methods]) for key in RELATIONS},
                "hit_token_limit_songs": sum(m["generation"]["hit_token_limit"] for m in methods),
                "phrase_count_mismatch_songs": sum(
                    len(m["lyrics"]) != len(m["template"]) for m in methods
                ),
                "pronunciation_failures": sum(m["pronunciation_failures"] for m in methods),
            }
            for key in ("reference_perplexity", "generated_perplexity"):
                stats = [m[key] for m in methods if m[key].get("token_count", 0) > 0]
                summary[key] = {
                    **(aggregate_perplexity(stats) if stats else {"perplexity": None}),
                    "scored_songs": len(stats),
                }
            return summary

        summaries[name] = {
            "predictor": getattr(predictor, "metadata", {}),
            "failed_songs": len(results) - len(successful),
            **summarize(successful),
            "common_subset": summarize([r["methods"][name] for r in common]),
        }
    # Generation coverage and reference-PPL coverage can differ (e.g. target too long).
    ppl_common = [
        r
        for r in common
        if all(
            r["methods"][n]["reference_perplexity"].get("token_count", 0) > 0 for n in predictors
        )
    ]
    for name in predictors:
        stats = [r["methods"][name]["reference_perplexity"] for r in ppl_common]
        summaries[name]["common_reference_perplexity"] = {
            **(aggregate_perplexity(stats) if stats else {"perplexity": None}),
            "scored_songs": len(stats),
        }
    return summaries, [r["song_id"] for r in common], [r["song_id"] for r in ppl_common]


def evaluate_pipeline(
    songs,
    predictors,
    decoder,
    *,
    prosody_mode="actual",
    seed=1234,
    progress=False,
    on_song=None,
    **generation_options,
):
    """Reuse loaded predictors/decoder. References enter scoring only, after generation."""
    if prosody_mode not in ("actual", "paper") or not predictors:
        raise ValueError("Select predictors and actual or paper prosody rules")
    for predictor in predictors.values():
        mode = getattr(predictor, "prosody_mode", prosody_mode)
        if mode != prosody_mode:
            raise ValueError("Bridge IPA rules differ from --prosody-mode")
        if mode != "actual" and isinstance(predictor, BridgePredictor):
            raise ValueError("Bridge IPA rules differ from the template decoder's rules")
    settings = decoder.generation_settings(**generation_options)
    results = []
    for song in tqdm(songs, desc="Pipeline evaluation", unit="song", disable=not progress):
        # Stable across filtering/reordering, with identical RNG initialization for each method.
        song_seed = (seed + int(hashlib.sha256(song.song_id.encode()).hexdigest()[:8], 16)) % 2**32
        result = {
            "song_id": song.song_id,
            "title": song.title,
            "midi": str(song.midi),
            "lyrics": str(song.lyrics),
            "midi_sha256": file_sha256(song.midi),
            "lyrics_sha256": file_sha256(song.lyrics),
            "seed": song_seed,
            "methods": {},
        }
        try:
            references, removed = read_references(song.lyrics, prosody_mode=prosody_mode)
        except ValueError as exc:
            result["reference_error"] = str(exc)
            result["methods"] = {
                name: {"error": str(exc), "stage": "reference"} for name in predictors
            }
        else:
            result.update(reference=references, ignored_numeric_markers=removed)
            for name, predictor in predictors.items():
                stage = "template"
                try:
                    records = predictor(song)
                    stage = "generation"
                    decoded = decoder.generate(
                        records,
                        title=song.title,
                        prosody_mode=prosody_mode,
                        seed=song_seed,
                        **settings,
                    )
                except ValueError as exc:
                    result["methods"][name] = {"error": str(exc), "stage": stage}
                    continue
                # evaluate_prosody has already pronounced each generated phrase once.
                generated = [
                    {
                        "text": p["text"],
                        "syllables": p["generated_syllables"] or [],
                        "error": p["error"],
                    }
                    for p in decoded["metrics"]["phrases"][: len(decoded["lyrics"])]
                ]
                try:
                    reference_ppl = decoder.score_reference(
                        decoded["encoded_source"],
                        [r["text"] for r in references],
                        prosody_mode=prosody_mode,
                    )
                except ValueError as exc:
                    reference_ppl = {"perplexity": None, "error": str(exc)}
                result["methods"][name] = {
                    "template": records,
                    "lyrics": decoded["lyrics"],
                    "generated_prosody": generated,
                    "encoded_source": decoded["encoded_source"],
                    "generated_token_ids": decoded["generated_token_ids"],
                    "generation": decoded["generation"],
                    "pronunciation_failures": decoded["metrics"]["pronunciation_failures"],
                    "template_reference": compare_prosody(records, references),
                    "lyrics_template": compare_prosody(generated, records),
                    "lyrics_reference": compare_prosody(generated, references),
                    "generated_perplexity": decoded["generated_perplexity"],
                    "reference_perplexity": reference_ppl,
                }
        results.append(result)
        if on_song is not None:
            on_song(result)
    methods, common, ppl_common = summarize_methods(results, predictors)
    return {
        "schema_version": 1,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "unit": "whole_song",
        "prosody_mode": prosody_mode,
        "reference_rules": REPO_RULES if prosody_mode == "actual" else PAPER_RULES,
        "metric_version": ACTUAL_METRIC_VERSION if prosody_mode == "actual" else METRIC_VERSION,
        "generation": {"seed": seed, **settings},
        "songs": len(results),
        "common_song_ids": common,
        "common_reference_perplexity_song_ids": ppl_common,
        "metric_definitions": {
            "template_reference": "Predicted input template versus reference lyric IPA",
            "lyrics_template": "Generated lyric IPA versus its actual BART input template",
            "lyrics_reference": "Generated lyric IPA versus reference lyric IPA (full pipeline)",
            "accuracy": "Micro positional stress/length/joint accuracy; missing/extra slots wrong",
            "prosody_bleu": "Phrase macro mean, unsmoothed joint BLEU-4; short/failed phrases zero",
            "counts": "Pair by phrase, then slot order; no realignment or forced counts",
            "reference_perplexity": "Teacher-forced reference BPE NLL under each predicted source",
            "generated_perplexity": "Generated text conditional self-score; not external fluency",
            "perplexity_aggregation": (
                "exp(sum(NLL) / sum(scored tokens)); BOS/PAD excluded, EOS included"
            ),
            "coverage": "Failures listed and excluded; common_subset uses the same songs",
            "ppl_coverage": "common_reference_perplexity uses songs scored under every method",
            "phrasing": "All methods retain complete MIDI phrases, including unmarked tails",
            "reference_usage": "Cleaned lyrics supply scoring targets only; no overlap exclusion",
            "seed": "(base seed + first 32 bits of SHA256(song_id)) modulo 2**32, reset per method",
        },
        "methods": methods,
        "per_song": results,
    }


def markdown_summary(report):
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "# Full pipeline evaluation",
        "",
        f"Songs: {report['songs']}; common generation subset: {len(report['common_song_ids'])}; "
        f"IPA rules: {report['prosody_mode']}. Scores below use the common subset.",
        "",
        "| Input | Songs | Input/ref BLEU | Lyrics/input BLEU | Lyrics/ref BLEU | "
        "Lyrics/ref joint accuracy | Reference PPL | Self PPL | Token-limit songs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        score = method["common_subset"]
        lines.append(
            f"| {name} | {score['evaluated_songs']} | "
            + " | ".join(fmt(score[k]["prosody_bleu"]) for k in RELATIONS)
            + f" | {fmt(score['lyrics_reference']['accuracy']['combined'])} | "
            + f"{fmt(method['common_reference_perplexity']['perplexity'])} | "
            + f"{fmt(score['generated_perplexity']['perplexity'])} | "
            + f"{score['hit_token_limit_songs']} |"
        )
    lines += [
        "",
        "BLEU uses joint stress/length symbols on a 0–1 scale. Missing/extra phrases and "
        "failed pronunciations score zero. Positional accuracy also penalizes missing/extra slots.",
        "",
        f"Reference PPL uses {len(report['common_reference_perplexity_song_ids'])} songs scored "
        "by every method, with token-weighted NLL. Self PPL measures model confidence in its "
        "own output. Neither score is an external fluency measure.",
        "",
        "## Coverage",
        "",
    ]
    for name, method in report["methods"].items():
        lines.append(
            f"- {name}: {method['evaluated_songs']}/{report['songs']} generated; "
            f"{method['reference_perplexity']['scored_songs']} reference PPL scored; "
            f"{method['phrase_count_mismatch_songs']} phrase-count mismatches; "
            f"{method['pronunciation_failures']} pronunciation failures."
        )
    for song in report["per_song"]:
        for name, method in song["methods"].items():
            if "error" in method:
                lines.append(f"- {song['song_id']} [{name}], {method['stage']}: {method['error']}")
            elif "error" in method["reference_perplexity"]:
                lines.append(
                    f"- {song['song_id']} [{name}], reference PPL: "
                    + method["reference_perplexity"]["error"]
                )
    if report.get("readable_reports"):
        lines += [
            "",
            f"[Detailed song reports and lyrics](<{report['readable_reports']['index']}>)",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="XAI directory or .pt file")
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/bridge-eval"))
    parser.add_argument(
        "--methods", nargs="+", choices=(*MIDI_METHODS, "bridge"), default=["actual", "bridge"]
    )
    parser.add_argument("--prosody-mode", choices=("actual", "paper"), default="actual")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--track", type=int, default=0)
    parser.add_argument("--songs", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--top-k", type=int, default=1, help="Default: reproducible greedy decoding"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--bart-config")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New .json report path")
    parser.add_argument(
        "--from-report",
        type=Path,
        help="Export readable reports from saved pipeline JSON; no model loading",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--report-count",
        type=int,
        default=3,
        metavar="K",
        help="Detailed reports for K random evaluated songs (default 3; 0 disables)",
    )
    selection.add_argument(
        "--report-songs",
        nargs="+",
        metavar="SONG",
        help="Detailed reports for these song IDs instead of a random sample",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="New report folder (default: OUTPUT.details or FROM_REPORT.details)",
    )
    parser.add_argument(
        "--report-seed",
        type=int,
        help="Random report selection seed (default: saved evaluation seed)",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    if args.report_count < 0:
        parser.error("--report-count must be nonnegative")
    source = args.from_report or args.output
    if source is None:
        parser.error(
            "--output is required for evaluation; use --from-report to export saved results"
        )
    report_dir = args.report_dir or source.with_suffix(".details")
    if (args.report_count or args.report_songs) and report_dir.exists():
        parser.error("--report-dir must be a new directory")
    if args.from_report:
        if args.checkpoint or args.bridge_checkpoint or args.output or args.songs:
            parser.error(
                "--from-report does not use checkpoints, --output or --songs; "
                "use --report-songs to select saved songs"
            )
        report = json.loads(args.from_report.read_text(encoding="utf-8"))
        try:
            report["readable_reports"] = export_readable_reports(
                report,
                report_dir,
                count=args.report_count,
                songs=args.report_songs,
                seed=args.report_seed,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"Readable reports: {report['readable_reports']['index']}"
            if report["readable_reports"]
            else "Readable reports disabled (--report-count 0)"
        )
        return report
    if args.checkpoint is None:
        parser.error("--checkpoint is required for evaluation")
    if "bridge" in args.methods and args.bridge_checkpoint is None:
        parser.error("--methods bridge requires --bridge-checkpoint")
    if args.bridge_checkpoint and "bridge" not in args.methods:
        parser.error("--bridge-checkpoint requires bridge in --methods")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    markdown = args.output.with_suffix(".md")
    journal = args.output.with_suffix(".songs.jsonl")
    if any(path.exists() for path in (args.output, markdown, journal)):
        parser.error("Report paths must be new (.json, .md, .songs.jsonl)")
    songs = discover_songs(args.data_dir, songs=args.songs, limit=args.limit)
    if args.report_songs:
        unknown = set(args.report_songs) - {song.song_id for song in songs}
        if unknown:
            parser.error(
                "--report-songs must belong to the evaluated set: " + ", ".join(sorted(unknown))
            )
    predictors = {
        name: MidiPredictor(name, track=args.track) for name in args.methods if name != "bridge"
    }
    if "bridge" in args.methods:
        predictors["bridge"] = BridgePredictor(
            args.bridge_checkpoint,
            track=args.track,
            device=args.device,
        )
    decoder = TemplateDecoder(
        args.checkpoint,
        device=args.device,
        tokenizer_dir=args.tokenizer_dir,
        bart_config=args.bart_config,
        local_files_only=args.local_files_only,
    )
    options = {
        key: getattr(args, key)
        for key in (
            "top_k",
            "temperature",
            "max_new_tokens",
            "no_repeat_ngram_size",
            "repetition_penalty",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Flush each completed song so long generation jobs retain inspectable results on interruption.
    with journal.open("x", encoding="utf-8") as handle:

        def save_song(song):
            handle.write(json.dumps(song, allow_nan=False) + "\n")
            handle.flush()

        report = evaluate_pipeline(
            songs,
            predictors,
            decoder,
            prosody_mode=args.prosody_mode,
            seed=args.seed,
            progress=not args.no_progress,
            on_song=save_song,
            **options,
        )
    weights = args.checkpoint if args.checkpoint.is_file() else args.checkpoint / "weights.pt"
    report.update(
        data_dir=str(args.data_dir.resolve()),
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_weights_sha256=file_sha256(weights),
        checkpoint_run=decoder.run,
        device=str(decoder.device),
        software_versions=package_versions(),
    )
    report["readable_reports"] = export_readable_reports(
        report,
        report_dir,
        count=args.report_count,
        songs=args.report_songs,
        seed=args.report_seed,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = markdown_summary(report)
    markdown.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Detailed report: {args.output}")
    return report


if __name__ == "__main__":
    main()
