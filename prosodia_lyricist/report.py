"""Readable and machine-readable MIDI inference reports."""

import json
from pathlib import Path


def pattern(syllables):
    return " ".join(f"<{s['stress']},{s['length']}>" for s in syllables)


def markdown_report(report):
    def escaped(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "'")

    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"

    rows = [
        f"# {escaped(report['title'] or 'Untitled')} — MIDI sanity check",
        "",
        f"- Input: `{report['midi']}`",
        f"- Unit: **whole song**, {len(report['template'])} ordered phrases in one encoder input.",
        f"- Template method: `{report['stress_source']}`; one note per syllable.",
        "- MIDI tick zero anchors the metrical grid; phrase markers are inclusive of note onsets.",
    ]
    if report.get("checkpoint"):
        rows.append(f"- Checkpoint: `{report['checkpoint']}`")
    for warning in report.get("warnings", []):
        rows.append(f"- **Note:** {warning}")
    metrics = report.get("metrics")
    if metrics:
        rows.extend(
            [
                "",
                "## Scores",
                "",
                f"- Prosody-BLEU-4 (phrase mean): **{number(metrics['prosody_bleu'])}**",
                f"- Input / generated phrases: {metrics['input_phrases']} / "
                f"{metrics['generated_phrases']}",
                f"- Pronunciation failures: {metrics['pronunciation_failures']}",
            ]
        )
        for key, label in (
            ("generated_perplexity", "Generated-sequence conditional perplexity (self-score)"),
            ("reference_perplexity", "Reference conditional perplexity"),
        ):
            if report.get(key):
                value = report[key]
                rows.append(
                    f"- {label}: **{number(value['perplexity'])}** "
                    f"({value['token_count']} scored BPE tokens)"
                )
        rows.extend(
            [
                "",
                "BLEU uses atomic strength/length pairs, fixed order 4, no smoothing or BOS/EOS. "
                "Missing/extra/IPA-failed phrases score zero; phrases shorter than four "
                "syllables can score zero even when identical. Self-perplexity is a diagnostic, "
                "not a held-out reference score. See docs/evaluation.md.",
                "",
                "## Generated lyrics",
                "",
            ]
        )
        rows.extend(f"{i + 1}. {escaped(line)}" for i, line in enumerate(report["lyrics"]))
    count = max(len(report["template"]), len(report.get("lyrics", [])))
    for index in range(count):
        source = report["template"][index] if index < len(report["template"]) else None
        entry = metrics["phrases"][index] if metrics else None
        references = report.get("reference_lines", [])
        reference_labels = report.get("reference_syllables", [])
        reference = reference_labels[index] if index < len(reference_labels) else []
        rows.extend(["", f"## Phrase {index + 1}", ""])
        if source:
            rows.extend(
                [
                    f"Input: **{len(source['syllables'])} syllables**. "
                    f"Long means duration > {source['length_threshold_ticks']:.3f} ticks.",
                    "",
                    f"`{pattern(source['syllables'])}`",
                ]
            )
        else:
            rows.append("**Extra generated phrase; no corresponding input phrase.**")
        if entry:
            rows.extend(
                [
                    "",
                    f"Lyrics: {escaped(entry['text']) or '(missing)'}",
                    "",
                    f"Output syllables: {entry['generated_count']}; "
                    f"prosody-BLEU: {number(entry['prosody_bleu'])}.",
                ]
            )
            if entry["error"]:
                rows.append(f"**Scoring issue:** {escaped(entry['error'])}")
            else:
                rows.extend(["", f"`{pattern(entry['generated_syllables'])}`"])
        if index < len(references):
            rows.extend(["", f"Ground-truth lyrics: {escaped(references[index])}"])
            if index < len(reference_labels):
                rows.extend([
                    "",
                    f"Ground-truth prosody (derived from lyric IPA): {len(reference)} syllables.",
                    "",
                    f"`{pattern(reference)}`",
                ])
        reference_columns = bool(references)
        rows.extend(
            [
                "",
                "Position-by-position inspection only; this is not a fitted lyric/note alignment.",
                "",
                "| Slot | Note | Start–end ticks | Bar:beat | Input | Word / IPA | Output |"
                + (
                    " Ground-truth word / IPA | Ground-truth prosody |"
                    if reference_columns else ""
                ),
                "| --- | --- | --- | --- | --- | --- | --- |"
                + (" --- | --- |" if reference_columns else ""),
            ]
        )
        syllables = source["syllables"] if source else []
        generated = (entry["generated_syllables"] or []) if entry else []
        for slot in range(max(len(syllables), len(generated), len(reference))):
            inp = syllables[slot] if slot < len(syllables) else None
            out = generated[slot] if slot < len(generated) else None
            note = inp["note"] if inp else None
            cells = [
                slot + 1,
                note["pitch"] if note else "—",
                f"{note['start_tick']}–{note['end_tick']}" if note else "—",
                f"{note['measure']}:{note['quarter_beat']:g}" if note else "—",
                pattern([inp]) if inp else "—",
                f"{out['word']} / {out['ipa']}" if out else "—",
                pattern([out]) if out else "—",
            ]
            if reference_columns:
                truth = reference[slot] if slot < len(reference) else None
                cells.extend([
                    f"{truth['word']} / {truth['ipa']}" if truth else "—",
                    pattern([truth]) if truth else "—",
                ])
            rows.append("| " + " | ".join(escaped(c) for c in cells) + " |")
    return "\n".join(rows) + "\n"


def write_report(report, prefix):
    prefix = Path(prefix)
    paths = [Path(str(prefix) + suffix) for suffix in (".json", ".md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Report already exists; choose a new --report-prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    contents = [json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"]
    contents.append(markdown_report(report))
    for path, content in zip(paths, contents):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
