"""Reproducible, read-only audit of prepared bridge data and optional raw DALI.

Run: python -m prosodia_lyricist.audit_bridge --raw-dir DALI_v2 --plot
Outputs go to outputs/dali-bridge-audit; no preparation or model inference is run.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .dali import read_annotation

METRICS = ("notes", "ipa", "sung")
BINS = ((1, 2), (3, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 40), (41, 64), (65, 99999))
BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")
DECORATIONS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|\*[^*]+\*")
KEYWORDS = re.compile(
    r"\b(?:chorus|refrain|instrumental|solo|intro|outro|interlude|verse|bridge)\b", re.I
)
# Exact label-shaped content, not a keyword anywhere in sung lyrics.
LABEL = re.compile(
    r"(?:(?:repeat|reprise)\s+)?"
    r"(?P<kind>(?:pre\s*|post\s*)?chorus|refrain|instrumental(?:\s+(?:break|solo))?|"
    r"(?:(?:guitar|piano|drum|drums|bass|sax|saxophone|keyboard|violin|organ|harmonica)\s+)?"
    r"solo|intro|outro|interlude|verse|bridge)"
    r"(?:\s+(?:\d+|[ivx]+|x\s*\d+|\d+\s*x|repeat|reprise))*",
    re.I,
)


def label_kind(text):
    clean = re.sub(r"[-_:]+", " ", text.lower()).strip(" \t.!*")
    clean = " ".join(clean.split())
    match = LABEL.fullmatch(clean)
    if not match:
        return None
    kind = match["kind"]
    if "chorus" in kind or kind == "refrain":
        return "chorus_or_refrain"
    if "instrumental" in kind or "solo" in kind:
        return "instrumental_or_solo"
    return "other_section"


def marker_flags(text):
    spans = list(DECORATIONS.finditer(text))
    labels = [(m, label_kind(m[0][1:-1])) for m in spans]
    labels = [(m, kind) for m, kind in labels if kind]
    remaining = text
    for match, _ in reversed(labels):
        remaining = remaining[: match.start()] + remaining[match.end() :]
    standalone = bool(labels) and not re.search(r"\w", remaining)
    bare = label_kind(text) if not spans else None
    return {
        "bracketed_label": any(m[0][0] != "*" for m, _ in labels),
        "decorated_label": bool(labels),
        "standalone_label": standalone or bool(bare),
        "inline_label": bool(labels) and not standalone,
        "bare_label_candidate": bool(bare),
        "kinds": sorted({kind for _, kind in labels} | ({bare} if bare else set())),
        "has_brackets": bool(BRACKETS.search(text)),
        "keyword_mention": bool(KEYWORDS.search(text)),
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    result = {"n": len(a), "sum": float(a.sum()), "mean": float(a.mean())}
    for name, q in (
        ("min", 0),
        ("p5", 5),
        ("p25", 25),
        ("median", 50),
        ("p75", 75),
        ("p90", 90),
        ("p95", 95),
        ("p99", 99),
        ("max", 100),
    ):
        result[name] = float(np.percentile(a, q))
    return result


def fraction(rows, selected):
    return {
        "rows": len(selected),
        "pct_rows": 100 * len(selected) / len(rows) if rows else 0,
        "songs": len({r["song_id"] for r in selected}),
        "pct_songs": 100 * len({r["song_id"] for r in selected}) / len({r["song_id"] for r in rows})
        if rows
        else 0,
    }


def summarize(rows):
    result = {"rows": len(rows), "songs": len({r["song_id"] for r in rows})}
    result["lengths"] = {m: describe([r[m] for r in rows]) for m in METRICS}
    result["histograms"] = {m: dict(sorted(Counter(r[m] for r in rows).items())) for m in METRICS}
    result["thresholds"] = {}
    for metric in METRICS:
        thresholds = {}
        for threshold in (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 64):
            for op in ("le", "gt"):
                selected = [r for r in rows if (r[metric] <= threshold) == (op == "le")]
                thresholds[f"{op}_{threshold}"] = fraction(rows, selected)
        result["thresholds"][metric] = thresholds
    result["mismatches"] = {}
    for left, right in (("ipa", "notes"), ("sung", "notes"), ("ipa", "sung")):
        diff = np.array([r[left] - r[right] for r in rows])
        details = {
            op: fraction(rows, [r for r, d in zip(rows, diff) if predicate(d)])
            for op, predicate in (
                ("less", lambda d: d < 0),
                ("equal", lambda d: d == 0),
                ("greater", lambda d: d > 0),
                ("different", lambda d: d != 0),
                ("abs_ge_3", lambda d: abs(d) >= 3),
            )
        }
        details["mean_signed_difference"] = float(diff.mean()) if len(diff) else None
        details["mean_absolute_difference"] = float(abs(diff).mean()) if len(diff) else None
        result["mismatches"][f"{left}_minus_{right}"] = details
    result["duration_seconds"] = describe([r["duration_seconds"] for r in rows])
    return result


def conditional_counts(rows, unit):
    result = []
    for lo, hi in BINS:
        subset = [r for r in rows if lo <= r["notes"] <= hi]
        if not subset:
            continue
        ipa = describe([r["ipa"] for r in subset])
        result.append(
            {
                "unit": unit,
                "notes_bin": f"{lo}-{hi}" if hi < 99999 else f"{lo}+",
                **fraction(rows, subset),
                "ipa_mean": ipa["mean"],
                "ipa_median": ipa["median"],
                "ipa_p10": float(np.percentile([r["ipa"] for r in subset], 10)),
                "ipa_p90": ipa["p90"],
                "mean_ipa_per_note": float(np.mean([r["ipa"] / r["notes"] for r in subset])),
                "pct_ipa_less_than_notes": 100
                * sum(r["ipa"] < r["notes"] for r in subset)
                / len(subset),
            }
        )
    return result


def baseline_results(lines, max_syllables):
    """Fit simple count-only baselines on train; evaluate on untouched splits."""
    train = [r for r in lines if r["split"] == "train"]
    by_note = defaultdict(list)
    for row in train:
        by_note[row["notes"]].append(row["ipa"])
    if not by_note:
        return []
    mode = Counter(r["ipa"] for r in train).most_common(1)[0][0]
    note_modes = {n: Counter(values).most_common(1)[0][0] for n, values in by_note.items()}
    result = []
    for split in ("valid", "test"):
        subset = [r for r in lines if r["split"] == split]
        for name in ("train_global_mode", "notes_clipped_to_target_limit", "train_note_count_mode"):
            for lo, hi in ((1, 99999), *BINS):
                group = [r for r in subset if lo <= r["notes"] <= hi]
                if not group:
                    continue
                preds = []
                for row in group:
                    n = row["notes"]
                    nearest = min(by_note, key=lambda k: (abs(k - n), k))
                    pred = (
                        mode
                        if name == "train_global_mode"
                        else min(n, max_syllables)
                        if name == "notes_clipped_to_target_limit"
                        else note_modes[nearest]
                    )
                    preds.append(pred)
                err = np.array(preds) - np.array([r["ipa"] for r in group])
                result.append(
                    {
                        "split": split,
                        "baseline": name,
                        "notes_bin": "all" if (lo, hi) == (1, 99999) else f"{lo}-{hi}",
                        "n": len(group),
                        "mae": float(abs(err).mean()),
                        "signed_error_pred_minus_ipa": float(err.mean()),
                        "exact_pct": float(100 * (err == 0).mean()),
                    }
                )
    return result


def scan_raw(directory, prepared_ids, output):
    counts = Counter()
    candidates, errors = [], []
    for path in sorted(directory.glob("*.gz")):
        counts["files"] += 1
        try:
            info, annot = read_annotation(path)
        except Exception as exc:  # Surface every failure; never silently lower the denominator.
            errors.append({"file": path.name, "error": str(exc)})
            continue
        sid = str(info["id"])
        language = str(info.get("metadata", {}).get("language", "")).lower()
        counts["readable_songs"] += 1
        counts[f"language_{language}"] += 1
        counts["prepared_songs_found"] += sid in prepared_ids
        for level in ("lines", "paragraphs"):
            for index, item in enumerate(annot.get(level, [])):
                text = str(item.get("text", ""))
                flags = marker_flags(text)
                if not (
                    flags["has_brackets"]
                    or flags["keyword_mention"]
                    or flags["standalone_label"]
                    or flags["inline_label"]
                ):
                    continue
                candidates.append(
                    {
                        "song_id": sid,
                        "title": info.get("title", ""),
                        "language": language,
                        "prepared": sid in prepared_ids,
                        "level": level,
                        "index": index,
                        "text": text,
                        **{**flags, "kinds": "|".join(flags["kinds"])},
                    }
                )
    write_csv(output / "raw_marker_candidates.csv", candidates)
    result = {"counts": dict(counts), "errors": errors, "groups": {}}
    for group in ("all_raw", "english_raw", "prepared_raw"):
        for level in ("lines", "paragraphs"):
            rows = [
                r
                for r in candidates
                if r["level"] == level
                and (
                    group == "all_raw"
                    or group == "english_raw"
                    and r["language"] == "english"
                    or group == "prepared_raw"
                    and r["prepared"]
                )
            ]
            stats = {}
            for flag in (
                "bracketed_label",
                "decorated_label",
                "standalone_label",
                "inline_label",
                "bare_label_candidate",
                "has_brackets",
                "keyword_mention",
            ):
                selected = [r for r in rows if r[flag]]
                stats[flag] = {
                    "rows": len(selected),
                    "songs": len({r["song_id"] for r in selected}),
                }
            for kind in ("chorus_or_refrain", "instrumental_or_solo", "other_section"):
                selected = [r for r in rows if kind in r["kinds"].split("|")]
                stats[kind] = {
                    "rows": len(selected),
                    "songs": len({r["song_id"] for r in selected}),
                    "standalone_rows": sum(r["standalone_label"] for r in selected),
                    "inline_rows": sum(r["inline_label"] for r in selected),
                }
            result["groups"][f"{group}_{level}"] = stats
    return result


def make_plot(lines, windows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [r for r in lines if r["split"] == "train"]
    pairs = [r for r in windows if r["split"] == "train" and r["line_count"] == 2]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"notes": "#277da1", "ipa": "#f3722c", "sung": "#43aa8b"}
    for ax, rows, title, xmax in (
        (axes[0, 0], train, "Individual lines", 30),
        (axes[0, 1], pairs, "Non-overlapping two-line windows", 50),
    ):
        for metric in METRICS:
            values = np.array([r[metric] for r in rows])
            ax.hist(
                values,
                bins=np.arange(0.5, xmax + 1.5),
                weights=np.ones(len(rows)) * 100 / len(rows),
                histtype="step",
                linewidth=1.7,
                label=metric.upper(),
                color=colors[metric],
            )
        ax.set(
            xlabel="Count (right tail beyond axis omitted)",
            ylabel="% of training examples",
            title=title,
            xlim=(0, xmax),
        )
        ax.legend(frameon=False)
    ax = axes[1, 0]
    matrix = np.zeros((40, 30))
    for r in train:
        if 1 <= r["notes"] <= 30 and 1 <= r["ipa"] <= 40:
            matrix[r["ipa"] - 1, r["notes"] - 1] += 1
    im = ax.imshow(
        np.log10(matrix + 1),
        origin="lower",
        aspect="auto",
        extent=(0.5, 30.5, 0.5, 40.5),
        cmap="viridis",
    )
    ax.plot([1, 30], [1, 30], "w--", linewidth=1)
    ax.set(xlabel="Notes per line", ylabel="IPA target syllables", title="Training support")
    fig.colorbar(im, ax=ax, label="log10(lines + 1)")
    ax = axes[1, 1]
    x, median, low, high = [], [], [], []
    for n in range(1, 81):
        values = [r["ipa"] for r in train if r["notes"] == n]
        if len(values) >= 20:
            x.append(n)
            q = np.percentile(values, [10, 50, 90])
            low.append(q[0])
            median.append(q[1])
            high.append(q[2])
    ax.fill_between(x, low, high, color=colors["ipa"], alpha=0.22, label="10th–90th percentile")
    ax.plot(x, median, color=colors["ipa"], label="Median IPA target")
    ax.plot([1, 28], [1, 28], "--", color="gray", label="One syllable per note")
    ax.set(
        xlabel="Notes per line (at least 20 training lines)",
        ylabel="IPA syllables",
        title="Longer melodies often have fewer syllables than notes",
        xlim=(0, 28),
        ylim=(0, 30),
    )
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("DALI bridge: the count distribution seen during training", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/dali-bridge"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dali-bridge-audit"))
    parser.add_argument("--max-notes", type=int, default=512)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.max_notes < 1:
        parser.error("--max-notes must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    width = config.get("lines_per_window", 2)
    if width != 2 or config.get("stress_source") != "ipa" or not config.get("include_melody"):
        parser.error("This audit expects IPA bridge data with two-line windows and melody arrays")
    lines, windows, markers, songs = [], [], [], []
    integrity = Counter()
    hashes = {}
    for split in ("train", "valid", "test"):
        path = args.prepared_dir / f"{split}.jsonl"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                song = json.loads(raw)
                sid = song["song_id"]
                song_rows = []
                integrity["wrong_split"] += manifest["songs"][sid]["split"] != split
                for line in song["lines"]:
                    melody = line["melody"]
                    notes = len(melody["midi_pitches"])
                    sung = line["sung_syllables"]
                    flags = marker_flags(line["text"])
                    row = {
                        "split": split,
                        "song_id": sid,
                        "title": song["title"],
                        "line_id": line["line_id"],
                        "paragraph_id": line.get("paragraph_id"),
                        "text": line["text"],
                        "notes": notes,
                        "ipa": len(line["syllables"]),
                        "sung": len(sung),
                        "start": line["start"],
                        "end": line["end"],
                        "duration_seconds": line["end"] - line["start"],
                        "melismatic_syllables": sum(s["note_count"] > 1 for s in sung),
                        "max_notes_per_sung_syllable": max(s["note_count"] for s in sung),
                        "section_label_candidate": bool(flags["kinds"]),
                    }
                    integrity["note_array_length_mismatch"] += not (
                        notes
                        == len(melody["onset_seconds"])
                        == len(melody["note_duration_seconds"])
                    )
                    integrity["note_count_not_explained_by_sung"] += notes != sum(
                        s["note_count"] for s in sung
                    )
                    integrity["ipa_count_not_explained_by_words"] += row["ipa"] != sum(
                        w["syllable_count"] for w in line["words"]
                    )
                    integrity["invalid_target_count"] += (
                        not 1 <= row["ipa"] <= config["max_syllables"]
                    )
                    if flags["has_brackets"] or flags["keyword_mention"] or flags["kinds"]:
                        markers.append({**row, **{**flags, "kinds": "|".join(flags["kinds"])}})
                    song_rows.append(row)
                for offset in range(0, len(song_rows), width):
                    group = song_rows[offset : offset + width]
                    row = {
                        "split": split,
                        "song_id": sid,
                        "title": song["title"],
                        "offset": offset,
                        "line_ids": "|".join(str(r["line_id"]) for r in group),
                        "line_count": len(group),
                        **{m: sum(r[m] for r in group) for m in METRICS},
                        "duration_seconds": group[-1]["end"] - group[0]["start"],
                        "gap_seconds": group[-1]["start"] - group[0]["end"]
                        if len(group) == 2
                        else 0,
                        "crosses_paragraph": len({r["paragraph_id"] for r in group}) > 1,
                        "any_ipa_le_2": any(r["ipa"] <= 2 for r in group),
                        "both_ipa_le_2": len(group) == 2 and all(r["ipa"] <= 2 for r in group),
                        "any_line_mismatch": any(r["ipa"] != r["notes"] for r in group),
                        "section_label_candidate": any(r["section_label_candidate"] for r in group),
                    }
                    row["retained"] = row["notes"] <= args.max_notes
                    windows.append(row)
                lines.extend(song_rows)
                songs.append(
                    {
                        "split": split,
                        "song_id": sid,
                        "lines": len(song_rows),
                        **{m: sum(r[m] for r in song_rows) for m in METRICS},
                    }
                )
        hashes[path.name] = {
            "actual": digest.hexdigest(),
            "expected": manifest["sha256"][path.name],
        }
        if digest.hexdigest() != manifest["sha256"][path.name]:
            raise ValueError(f"Checksum mismatch: {path}")
        print(f"Read and verified {split}", flush=True)
    integrity["duplicate_song_ids"] = len(songs) - len({r["song_id"] for r in songs})
    write_csv(output / "lines.csv", lines)
    write_csv(output / "windows.csv", windows)
    write_csv(output / "prepared_marker_candidates.csv", markers)
    summary = {
        "provenance": {
            "prepared_dir": str(args.prepared_dir.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "config": config,
            "preparation_counts": manifest["counts"],
            "hashes": hashes,
            "window_max_notes": args.max_notes,
        },
        "definitions": {
            "ipa": "Prepared IPA target syllables",
            "sung": "DALI syllables after continuation-note merging",
            "notes": "All melody notes, including melisma continuations",
            "windows": "Non-overlapping pairs starting at offset 0; final single retained",
            "marker_scope": "Text labels only, not acoustic chorus/instrumental detection",
            "bare_labels": "Exact label-shaped text is a candidate, not confirmed metadata",
            "percentiles": "NumPy linear interpolation",
        },
        "integrity": dict(integrity),
        "groups": {},
    }
    conditional = []
    for split in ("all", "train", "valid", "test"):
        ls = [r for r in lines if split == "all" or r["split"] == split]
        ws = [r for r in windows if (split == "all" or r["split"] == split) and r["retained"]]
        pairs = [r for r in ws if r["line_count"] == 2]
        group = {
            "lines": summarize(ls),
            "two_line_windows": summarize(pairs),
            "all_training_windows": summarize(ws),
            "singleton_windows": sum(r["line_count"] == 1 for r in ws),
            "skipped_windows": sum(
                not r["retained"] for r in windows if split == "all" or r["split"] == split
            ),
            "song_line_counts": describe(
                [s["lines"] for s in songs if split == "all" or s["split"] == split]
            ),
            "two_line_diagnostics": {
                key: fraction(pairs, [r for r in pairs if r[key]])
                for key in (
                    "crosses_paragraph",
                    "any_ipa_le_2",
                    "both_ipa_le_2",
                    "any_line_mismatch",
                )
            },
            "melismatic_syllables": sum(r["melismatic_syllables"] for r in ls),
            "max_notes_per_sung_syllable": max(r["max_notes_per_sung_syllable"] for r in ls),
        }
        group["two_line_gap_seconds"] = describe([r["gap_seconds"] for r in pairs])
        group["cleanup_impact"] = {
            key: fraction(ws, [r for r in ws if predicate(r)])
            for key, predicate in (
                ("drop_any_ipa_le_2", lambda r: r["any_ipa_le_2"]),
                ("drop_total_ipa_le_4", lambda r: r["ipa"] <= 4),
                ("drop_any_line_note_ipa_mismatch", lambda r: r["any_line_mismatch"]),
                ("drop_section_label_candidates", lambda r: r["section_label_candidate"]),
            )
        }
        summary["groups"][split] = group
        for unit, rows in (("lines", ls), ("two_line_windows", pairs)):
            conditional.extend({"split": split, **r} for r in conditional_counts(rows, unit))
    write_csv(output / "conditional_counts.csv", conditional)
    write_csv(output / "count_baselines.csv", baseline_results(lines, config["max_syllables"]))
    rejects = [json.loads(row) for row in (args.prepared_dir / "rejected.jsonl").open()]
    summary["rejections"] = {
        "rows": len(rejects),
        "reasons": dict(Counter(r["reason"] for r in rejects)),
    }
    if args.raw_dir:
        if not args.raw_dir.is_dir() or not any(args.raw_dir.glob("*.gz")):
            parser.error("--raw-dir must contain DALI .gz files")
        print("Scanning raw DALI line and paragraph text", flush=True)
        summary["raw_markers"] = scan_raw(args.raw_dir, {s["song_id"] for s in songs}, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        make_plot(lines, windows, output / "length_distributions.png")
    print(f"Wrote audit to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
