"""Score MIDI algorithms and learned bridges against the same lyric IPA references."""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from tqdm import tqdm

from .evaluation import text_prosody
from .ipa import PAPER_RULES, REPO_RULES
from .midi import midi_records

MIDI_METHODS = ("actual", "supplement", "heuristic")
METRICS = ("strength", "length", "combined")
NUMBER_MARKER = re.compile(r"\(\s*\d+\s*\)")


@dataclass(frozen=True)
class EvalSong:
    song_id: str
    midi: Path
    lyrics: Path

    @property
    def title(self):
        return self.song_id.replace("_", " ")


def discover_songs(directory, *, songs=None, limit=None):
    """Use actual MIDI/lyric pairs, since copied manifests can list absent songs."""
    directory = Path(directory).resolve()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    found = {path.stem: path for path in sorted(directory.glob("*.mid"))}
    if not found:
        raise ValueError(f"No .mid files in {directory}")
    selected = sorted(found) if songs is None else list(dict.fromkeys(songs))
    unknown = set(selected) - found.keys()
    if unknown:
        raise ValueError(f"Unknown songs: {', '.join(sorted(unknown))}")
    result = []
    for name in selected:
        lyrics = directory / f"{name}.lyrics.txt"
        if not lyrics.is_file():
            raise ValueError(f"Missing lyrics for {name}: {lyrics}")
        result.append(EvalSong(name, found[name], lyrics))
    if not result:
        raise ValueError("No songs selected")
    return result[:limit]


def read_references(path, *, prosody_mode):
    lines, removed = [], 0
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        text, count = NUMBER_MARKER.subn("", raw)
        removed += count
        text = text.strip()
        if text:
            try:
                syllables = text_prosody(text, prosody_mode=prosody_mode)
            except ValueError as exc:
                raise ValueError(f"Lyric line {line_number}: {exc}") from exc
            lines.append({"text": text, "syllables": syllables})
    if not lines:
        raise ValueError(f"No lyric phrases in {path}")
    return lines, removed


class MidiPredictor:
    def __init__(self, method, *, track=0):
        if method not in MIDI_METHODS:
            raise ValueError(f"Unknown MIDI method: {method}")
        self.method, self.track = method, track
        self.metadata = {"kind": "midi", "stress_source": method, "track": track}

    def __call__(self, song):
        return midi_records(
            song.midi, title=song.title, track=self.track, stress_source=self.method,
            phrase_policy="complete", max_syllables=float("inf"),
        )


class BridgePredictor:
    """Load once; inference sees MIDI only, never reference labels or counts."""

    def __init__(self, checkpoint, *, track=0, device="auto"):
        from .bridge_model import ProsodyBridge
        from .melody_checkpoint import file_sha256
        from .runtime import select_device

        self.checkpoint = Path(checkpoint).resolve()
        self.device = select_device(device)
        self.model = ProsodyBridge.load(self.checkpoint).to(self.device).eval()
        self.track = track
        self.prosody_mode = "actual" if self.model.prosody_rules == REPO_RULES else "paper"
        self.metadata = {
            "kind": "bridge", "checkpoint": str(self.checkpoint),
            "weights_sha256": file_sha256(self.checkpoint / "bridge_weights.pt"),
            "device": str(self.device), "track": track,
            "query_mode": self.model.query_mode, "bridge_scope": self.model.bridge_scope,
            "prosody_rules": self.model.prosody_rules, "slot_count_source": "notes",
        }

    def __call__(self, song):
        from .bridge_infer import predict_midi_templates

        records, _ = predict_midi_templates(
            self.model, song.midi, title=song.title, track=self.track,
            device=self.device, checkpoint=self.checkpoint, phrase_policy="complete",
        )
        return records


def _summary(counts):
    counts = dict(counts)
    return {
        "counts": counts,
        "accuracy": {
            name: counts.get(f"{name}_correct", 0) / counts["slots"]
            if counts.get("slots") else None
            for name in METRICS
        },
    }


def score_templates(predictions, references):
    """Pair by phrase/slot order; count every missing or extra position as incorrect."""
    totals, equal_totals, phrases = Counter(), Counter(), []
    for index, (prediction, reference) in enumerate(
        zip_longest(predictions, references), 1
    ):
        predicted = prediction["syllables"] if prediction is not None else []
        expected = reference["syllables"] if reference is not None else []
        counts = Counter({
            "slots": max(len(predicted), len(expected)),
            "reference_slots": len(expected), "predicted_slots": len(predicted),
            "paired_slots": min(len(predicted), len(expected)),
            "phrases": 1, "count_mismatch_phrases": int(len(predicted) != len(expected)),
            **{f"{name}_correct": 0 for name in METRICS},
        })
        for actual, gold in zip(predicted, expected):
            strength = actual["stress"] == gold["stress"]
            length = actual["length"] == gold["length"]
            counts.update(strength_correct=int(strength), length_correct=int(length),
                          combined_correct=int(strength and length))
        totals.update(counts)
        if len(predicted) == len(expected):
            equal_totals.update(counts)
        phrases.append({
            "phrase": index, "text": reference["text"] if reference is not None else None,
            **_summary(counts),
        })
    return {
        **_summary(totals), "equal_count_phrases": _summary(equal_totals), "phrases": phrases,
    }


def aggregate_scores(scores):
    totals, equal = Counter(), Counter()
    for score in scores:
        totals.update(score["counts"])
        equal.update(score["equal_count_phrases"]["counts"])
    return {**_summary(totals), "equal_count_phrases": _summary(equal)}


def evaluate_songs(songs, predictors, *, prosody_mode="actual", progress=False):
    """Predictors are callables: EvalSong -> phrase records containing syllables."""
    if prosody_mode not in ("actual", "paper"):
        raise ValueError("prosody_mode must be actual or paper")
    if not predictors:
        raise ValueError("Select at least one predictor")
    for predictor in predictors.values():
        mode = getattr(predictor, "prosody_mode", prosody_mode)
        if mode != prosody_mode:
            raise ValueError("Bridge IPA rules differ from --prosody-mode")
    results = []
    for song in tqdm(songs, desc="MIDI evaluation", unit="song", disable=not progress):
        result = {"song_id": song.song_id, "midi": str(song.midi), "lyrics": str(song.lyrics)}
        try:
            references, removed = read_references(song.lyrics, prosody_mode=prosody_mode)
        except ValueError as exc:
            results.append({**result, "reference_error": str(exc), "methods": {}})
            continue
        result.update(ignored_numeric_markers=removed, reference_phrases=len(references))
        methods = result["methods"] = {}
        for name, predictor in predictors.items():
            try:
                predictions = predictor(song)
            except ValueError as exc:
                methods[name] = {"error": str(exc)}
                continue
            methods[name] = score_templates(predictions, references)
        results.append(result)
    common = [
        result for result in results
        if all("accuracy" in result["methods"].get(name, {}) for name in predictors)
    ]
    reports = {}
    for name, predictor in predictors.items():
        successful = [r for r in results if "accuracy" in r["methods"].get(name, {})]
        reports[name] = {
            "predictor": getattr(predictor, "metadata", {}),
            "evaluated_songs": len(successful), "failed_songs": len(results) - len(successful),
            **aggregate_scores([r["methods"][name] for r in successful]),
            "common_subset": aggregate_scores([r["methods"][name] for r in common]),
        }
    return {
        "schema_version": 1, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "prosody_mode": prosody_mode,
        "reference_rules": REPO_RULES if prosody_mode == "actual" else PAPER_RULES,
        "metric_definitions": {
            "reference": "IPA from cleaned lyric text, not corpus stress annotations",
            "alignment": "Phrase order, then syllable position within each phrase; no realignment",
            "accuracy": "Micro accuracy; denominator is sum(max(predicted, reference) per phrase)",
            "combined": "Both strength and length correct at the same slot",
            "count_mismatches": "Missing/extra slots and missing/extra phrases count as incorrect",
            "equal_count_phrases": "Diagnostic restricted to phrases with equal slot counts",
            "failures": "Reported explicitly and excluded; compare common_subset for same songs",
            "phrasing": "Complete MIDI phrases including unmarked tail, for every method",
            "strength": "Exact labels; actual IPA substrong is not collapsed to strong",
        },
        "songs": len(results), "common_song_ids": [r["song_id"] for r in common],
        "methods": reports, "per_song": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/bridge-eval"))
    parser.add_argument("--methods", nargs="+", choices=(*MIDI_METHODS, "bridge"))
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--prosody-mode", choices=("actual", "paper"),
                        help="Default: bridge checkpoint's rules, otherwise actual")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--track", type=int, default=0)
    parser.add_argument("--songs", nargs="+", help="Song file stems to evaluate")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, help="Save detailed JSON; stdout shows a summary")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    methods = args.methods or [*MIDI_METHODS, *(["bridge"] if args.bridge_checkpoint else [])]
    if "bridge" in methods and args.bridge_checkpoint is None:
        parser.error("--methods bridge requires --bridge-checkpoint")
    if args.bridge_checkpoint and "bridge" not in methods:
        parser.error("--bridge-checkpoint requires bridge in --methods")
    songs = discover_songs(args.data_dir, songs=args.songs, limit=args.limit)
    predictors = {
        name: MidiPredictor(name, track=args.track) for name in methods if name != "bridge"
    }
    if "bridge" in methods:
        predictors["bridge"] = BridgePredictor(
            args.bridge_checkpoint, track=args.track, device=args.device,
        )
    mode = args.prosody_mode or getattr(predictors.get("bridge"), "prosody_mode", "actual")
    report = evaluate_songs(songs, predictors, prosody_mode=mode, progress=not args.no_progress)
    report["data_dir"] = str(args.data_dir.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"{report['songs']} songs; IPA rules: {mode}; "
          f"common subset: {len(report['common_song_ids'])}")
    print(f"{'Method':<14} {'Songs':>7} {'Strength':>10} {'Length':>10} {'Combined':>10}")
    for name, summary in report["methods"].items():
        values = [
            f"{value:.2%}" if value is not None else "n/a"
            for value in summary["accuracy"].values()
        ]
        print(f"{name:<14} {summary['evaluated_songs']:>3}/{report['songs']:<3} "
              + " ".join(f"{value:>10}" for value in values))
    for result in report["per_song"]:
        if "reference_error" in result:
            print(f"{result['song_id']}: reference error: {result['reference_error']}")
        for name, score in result["methods"].items():
            if "error" in score:
                print(f"{result['song_id']} [{name}]: {score['error']}")
    if args.output:
        print(f"Detailed report: {args.output}")
    return report


if __name__ == "__main__":
    main()
