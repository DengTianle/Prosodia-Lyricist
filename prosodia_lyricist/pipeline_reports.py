"""Readable pipeline samples, rendered from saved results without model or IPA calls."""

import random
from pathlib import Path
from urllib.parse import quote

from .evaluation import bleu_diagnostics
from .ipa import FALLBACK_SOURCE
from .report import markdown_report


def select_report_songs(report, *, count=3, songs=None, seed=None):
    if count < 0:
        raise ValueError("Report count must be nonnegative")
    rows = {row["song_id"]: row for row in report["per_song"]}
    if songs is not None:
        selected = list(dict.fromkeys(songs))
        unknown = set(selected) - rows.keys()
        if unknown:
            raise ValueError(f"Unknown report songs: {', '.join(sorted(unknown))}")
    else:
        eligible = sorted(
            name
            for name, row in rows.items()
            if any("error" not in method for method in row["methods"].values())
        )
        seed = report.get("generation", {}).get("seed", 1234) if seed is None else seed
        selected = random.Random(seed).sample(eligible, min(count, len(eligible)))
    return [rows[name] for name in selected]


def _component(value):
    if not value or value in (".", "..") or any(c in value for c in ("/", "\\", "\0")):
        raise ValueError(f"Unsafe report filename: {value!r}")
    return value


def _escaped(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _inference_report(report, song, name):
    """Adapt stored pipeline output to the same renderer used by infer."""
    method = song["methods"][name]
    predictor = report["methods"][name]["predictor"]
    learned = predictor.get("kind") == "bridge" or name == "bridge"
    source = method["template"]
    generated = method["generated_prosody"]
    phrases = []
    for index, score in enumerate(method["lyrics_template"]["phrases"]):
        syllables = source[index]["syllables"] if index < len(source) else []
        output = generated[index] if index < len(generated) else {}
        labels = output.get("syllables", [])
        error = output.get("error")
        phrases.append(
            {
                "text": method["lyrics"][index] if index < len(method["lyrics"]) else "",
                "generated_syllables": None if error else labels,
                "generated_count": None if error else len(labels),
                "prosody_bleu": score["prosody_bleu"],
                "bleu_ngram_counts": bleu_diagnostics(syllables, labels),
                "error": error,
                "pronunciation_fallback_words": sorted(
                    {s["word"] for s in labels if s.get("pronunciation_source") == FALLBACK_SOURCE}
                ),
            }
        )
    warnings = []
    if method["generation"]["hit_token_limit"]:
        warnings.append("Generation reached its token budget; output may be incomplete.")
    if len(source) != len(method["lyrics"]):
        warnings.append(
            "Generated phrase count differs from the input template. Ordinal pairing "
            "may misalign later phrases."
        )
    references = song.get("reference", [])
    if len(references) != len(source):
        warnings.append(
            "Reference phrase count differs from the input template. All reference "
            "phrases are shown, including those with no input or generated counterpart."
        )
    result = {
        "title": song["title"],
        "midi": song["midi"],
        "checkpoint": report.get("checkpoint"),
        "decoder_mode": "xai_single_stream",
        "stress_source": "learned" if learned else predictor.get("stress_source", name),
        "prosody_mode": report["prosody_mode"],
        "template": source,
        "generation": method["generation"],
        "warnings": warnings,
        "lyrics": method["lyrics"],
        "reference_lines": [r["text"] for r in references],
        "reference_syllables": [r["syllables"] for r in references],
        "generated_perplexity": method["generated_perplexity"],
        "metrics": {
            "prosody_bleu": method["lyrics_template"]["prosody_bleu"],
            "input_phrases": len(source),
            "generated_phrases": len(method["lyrics"]),
            "pronunciation_failures": method["pronunciation_failures"],
            "phrases": phrases,
        },
    }
    if method["reference_perplexity"].get("token_count", 0):
        result["reference_perplexity"] = method["reference_perplexity"]
    elif method["reference_perplexity"].get("error"):
        warnings.append(
            "Reference perplexity unavailable: " + method["reference_perplexity"]["error"]
        )
    if learned:
        result["bridge"] = {
            "checkpoint": predictor.get("checkpoint"),
            "count_sources": ["notes"] * len(source),
        }
        # Keep the Imagine report's musical comparison, using stored MIDI results.
        for other_name, other in song["methods"].items():
            metadata = report["methods"][other_name]["predictor"]
            if "error" not in other and metadata.get("stress_source") == "actual":
                result["note_comparison"] = {
                    "method": "actual",
                    "lines": other["template"],
                    "warning": None,
                }
                break
    return result


def detailed_markdown(report, song, name):
    method = song["methods"][name]
    if "error" in method:
        return (
            f"# {_escaped(song['title'])} — {_escaped(name)}\n\n"
            f"Evaluation failed at `{method['stage']}`: {_escaped(method['error'])}\n"
        )
    rendered = markdown_report(_inference_report(report, song, name))
    heading, rest = rendered.split("\n", 1)
    rows = [
        heading,
        "",
        f"Input method: **{_escaped(name)}**; song `{song['song_id']}`.",
        "",
        "## Pipeline comparison",
        "",
        "| Comparison | Prosody-BLEU | Strength accuracy | Length accuracy | Joint accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("template_reference", "Input / reference"),
        ("lyrics_template", "Generated / input"),
        ("lyrics_reference", "Generated / reference"),
    ):
        score = method[key]
        values = ["n/a" if score["prosody_bleu"] is None else f"{score['prosody_bleu']:.4f}"]
        values += [
            "n/a" if score["accuracy"][k] is None else f"{score['accuracy'][k]:.2%}"
            for k in ("strength", "length", "combined")
        ]
        rows.append(f"| {label} | " + " | ".join(values) + " |")
    rows += [
        "",
        "Phrase and syllable comparisons use ordinal positions; no fitted alignment. "
        "The detailed phrase scores below compare generated lyrics with the input template.",
        rest,
    ]
    return "\n".join(rows)


def export_readable_reports(report, directory, *, count=3, songs=None, seed=None):
    """Write a new report directory with one detailed Markdown/lyrics pair per method."""
    if songs is None and count == 0:
        return None
    selected = select_report_songs(report, count=count, songs=songs, seed=seed)
    directory = Path(directory).resolve()
    # Validate every path before writing anything, including when exporting a supplied JSON.
    for song in selected:
        _component(song["song_id"])
        for name in song["methods"]:
            _component(name)
    directory.mkdir(parents=True, exist_ok=False)
    seed = report.get("generation", {}).get("seed", 1234) if seed is None else seed
    rows = [
        "# Pipeline song reports",
        "",
        f"{len(selected)} songs; "
        + ("explicit selection." if songs is not None else f"random sample, seed `{seed}`."),
        "",
        "Each method uses the same evaluated song. Reports include generated and reference "
        "lyrics, input prosody, per-phrase counts, word/IPA inspection, and scores.",
        "",
    ]
    for song in selected:
        folder = directory / song["song_id"]
        folder.mkdir()
        prefix = quote(song["song_id"], safe="")
        rows += [f"## {_escaped(song['title'])}", ""]
        if song.get("reference"):
            (folder / "reference.lyrics.txt").write_text(
                "\n".join(r["text"] for r in song["reference"]) + "\n",
                encoding="utf-8",
            )
            rows.append(f"- [Reference lyrics]({prefix}/reference.lyrics.txt)")
        elif song.get("reference_error"):
            rows.append(f"- Reference error: {_escaped(song['reference_error'])}")
        for name, method in song["methods"].items():
            (folder / f"{name}.md").write_text(
                detailed_markdown(report, song, name), encoding="utf-8"
            )
            link = f"{prefix}/{quote(name, safe='')}"
            rows.append(
                f"- **{_escaped(name)}**: [Detailed report]({link}.md)"
                + (
                    " (evaluation failed)"
                    if "error" in method
                    else f" · [Generated lyrics]({link}.lyrics.txt)"
                )
            )
            if "error" not in method:
                (folder / f"{name}.lyrics.txt").write_text(
                    "\n".join(method["lyrics"]) + "\n",
                    encoding="utf-8",
                )
        rows.append("")
    index = directory / "index.md"
    index.write_text("\n".join(rows), encoding="utf-8")
    return {
        "directory": str(directory),
        "index": str(index),
        "song_ids": [song["song_id"] for song in selected],
        "selection": "explicit" if songs is not None else "random",
        "seed": seed,
    }
