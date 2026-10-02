"""Readable and machine-readable MIDI inference reports."""

import json
from pathlib import Path


def pattern(syllables):
    return " ".join(f"<{s['stress']},{s['length']}>" for s in syllables)


def markdown_report(report):
    learned = report["stress_source"] == "learned"
    count_sources = report.get("bridge", {}).get("count_sources", [])
    note_slots = learned and bool(count_sources) and all(s == "notes" for s in count_sources)

    def escaped(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "'")

    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"

    rows = [
        f"# {escaped(report['title'] or 'Untitled')} — MIDI sanity check",
        "",
        f"- Input: `{report['midi']}`",
        f"- Unit: **whole song**, {len(report['template'])} ordered phrases in one encoder input.",
        (
            "- Template method: learned stress and IPA vowel length; one prosody slot per note."
            if note_slots
            else "- Template method: learned IPA syllable counts, stress and vowel length."
            if learned
            else f"- Template method: `{report['stress_source']}`; one note per syllable."
        ),
        (
            "- Phrase markers are inclusive of note onsets."
            if learned
            else "- MIDI tick zero anchors the metrical grid; "
            "phrase markers are inclusive of note onsets."
        ),
    ]
    if learned:
        rows.append(
            "- Scores compare lyrics with the predicted template; "
            "this is not ground-truth accuracy."
        )
        if report.get("bridge"):
            rows.append(f"- Bridge checkpoint: `{report['bridge']['checkpoint']}`")
    if report.get("checkpoint"):
        rows.append(f"- Checkpoint: `{report['checkpoint']}`")
        if report.get("decoder_mode"):
            rows.append(f"- Decoder: `{report['decoder_mode']}`")
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
                if "event_perplexity" in value:
                    event = value["event_perplexity"]
                    rows.append(
                        f"  - Including word-end events: {number(event['perplexity'])} "
                        f"({event['token_count']} events); separate diagnostic."
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
    explanation = report.get("decoder_explanation")
    if explanation:
        words = explanation["words"]
        changed = sum(w["correction_applied"] for w in words)
        rows.extend(
            [
                "",
                "## Decoder prosody feedback",
                "",
                f"Correction enabled: {explanation['prosody_correction']}. "
                f"Changed labels for {changed} / {len(words)} words. "
                f"Natural EOS: {explanation['completed']}. "
                f"Discarded unfinished word: {escaped(explanation['truncated_word']) or '(none)'}.",
                "",
                "These are the decoder's word-level predictions and feedback. BLEU is scored "
                "separately from final text, not from these heads.",
                "",
                "| Word | Predicted count / stress / length | Feedback count / stress / length |",
                "| --- | --- | --- |",
            ]
        )

        def word_features(values):
            return (
                f"{values['syllables']} / "
                f"{ {1: 'strong', 2: 'weak'}.get(values['stress'], 'pad') } / "
                f"{ {1: 'long', 2: 'short'}.get(values['length'], 'pad') }"
            )

        for word in words:
            rows.append(
                f"| {escaped(word['text'])} | {word_features(word['predicted'])} | "
                f"{word_features(word['corrected'])} |"
            )
    references = report.get("reference_lines", [])
    reference_labels = report.get("reference_syllables", [])
    reference_columns = bool(references)
    count = max(len(report["template"]), len(report.get("lyrics", [])))
    for index in range(count):
        source = report["template"][index] if index < len(report["template"]) else None
        entry = metrics["phrases"][index] if metrics else None
        reference = reference_labels[index] if index < len(reference_labels) else []
        rows.extend(["", f"## Phrase {index + 1}", ""])
        if source:
            rows.extend(
                [
                    f"Input: **{len(source['syllables'])} syllables**. "
                    + (
                        ("Fixed from " if note_slots else "Predicted from ")
                        + f"{len(source['melody']['midi_pitches'])} notes; "
                        "length labels describe IPA vowels."
                        if learned
                        else f"Long means duration > {source['length_threshold_ticks']:.3f} ticks."
                    ),
                    "",
                    f"`{pattern(source['syllables'])}`",
                ]
            )
        else:
            rows.append("**Extra generated phrase; no corresponding input phrase.**")
        comparison = report.get("note_comparison")
        if learned and comparison and index < len(comparison["lines"]):
            musical = comparison["lines"][index]
            rows.extend(["", "### MIDI notes and beat-based comparison", ""])
            if comparison["method"]:
                rows.append(
                    "Old `supplement` formula: duration-dependent beat stress; "
                    f"long means duration > {musical['length_threshold_ticks']:.3f} ticks "
                    "(whole-melody mean). Comparison only; these labels do not enter the model "
                    "or replace its predicted template in scoring."
                )
            else:
                rows.append(comparison["warning"])
            rows.extend(
                [
                    "",
                    "Notes are listed in MIDI order, independently of predicted syllable slots.",
                    "",
                    "| Note # | Note | Start–end ticks | Bar:beat | Beat-based prosody |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for note_index, label in enumerate(musical["syllables"], 1):
                note = label["note"]
                position = (
                    f"{note['measure']}:{note['quarter_beat']:g}"
                    if note["measure"] is not None
                    else "—"
                )
                cells = [
                    note_index,
                    note["pitch"],
                    f"{note['start_tick']}–{note['end_tick']}",
                    position,
                    pattern([label]) if comparison["method"] else "—",
                ]
                rows.append("| " + " | ".join(escaped(c) for c in cells) + " |")
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
            if entry.get("bleu_ngram_counts"):
                counts = ", ".join(
                    f"{c['order']}-gram {c['matched']}/{c['total']}"
                    for c in entry["bleu_ngram_counts"]
                )
                rows.extend(["", f"Clipped n-gram matches / candidate counts: {counts}."])
        if index < len(references):
            rows.extend(["", f"Ground-truth lyrics: {escaped(references[index])}"])
            if index < len(reference_labels):
                rows.extend(
                    [
                        "",
                        "Ground-truth prosody (derived from lyric IPA): "
                        f"{len(reference)} syllables.",
                        "",
                        f"`{pattern(reference)}`",
                    ]
                )
        rows.extend(
            [
                "",
                "Position-by-position inspection only; this is not a fitted lyric/note alignment.",
                "",
                (
                    "| Slot | Input | Word / IPA | Output |"
                    if learned
                    else "| Slot | Note | Start–end ticks | Bar:beat | "
                    "Input | Word / IPA | Output |"
                )
                + (
                    " Ground-truth word / IPA | Ground-truth prosody |" if reference_columns else ""
                ),
                (
                    "| --- | --- | --- | --- |"
                    if learned
                    else "| --- | --- | --- | --- | --- | --- | --- |"
                )
                + (" --- | --- |" if reference_columns else ""),
            ]
        )
        syllables = source["syllables"] if source else []
        generated = (entry["generated_syllables"] or []) if entry else []
        for slot in range(max(len(syllables), len(generated), len(reference))):
            inp = syllables[slot] if slot < len(syllables) else None
            out = generated[slot] if slot < len(generated) else None
            note = inp.get("note") if inp else None
            cells = [
                slot + 1,
                *(
                    []
                    if learned
                    else [
                        note["pitch"] if note else "—",
                        f"{note['start_tick']}–{note['end_tick']}" if note else "—",
                        f"{note['measure']}:{note['quarter_beat']:g}" if note else "—",
                    ]
                ),
                pattern([inp]) if inp else "—",
                f"{out['word']} / {out['ipa']}" if out else "—",
                pattern([out]) if out else "—",
            ]
            if reference_columns:
                truth = reference[slot] if slot < len(reference) else None
                cells.extend(
                    [
                        f"{truth['word']} / {truth['ipa']}" if truth else "—",
                        pattern([truth]) if truth else "—",
                    ]
                )
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
