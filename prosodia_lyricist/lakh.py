"""Prepare calibrated Melistas/Lakh alignments with the shared DALI IPA/melisma rules."""

import gzip
import json
import math
from bisect import bisect_right
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
from itertools import groupby, islice
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm

from .bridge_alignment import ALIGNMENT_POLICIES, LAKH_ALIGNMENT_SCHEME, aligned_note_indices
from .dali import AnnotationError, extract_lines
from .ipa import HISTORICAL_POLICY, REPO_RULES, backend, historical_provenance
from .melody_checkpoint import file_sha256
from .prepare import SCHEMA_VERSION, normalized_identity, split_for

DATASET = "melistas_lakh_2021"
METER_PHASE = "midi_signature_change_origin_v1"


def read_records(path, limit=None):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for row in islice(handle, limit):
            yield json.loads(row)


def lyric_text(record):
    """Rejoin spelling fragments within the authors' explicit word boundaries."""
    return " ".join(
        "".join(n.get("syllable_text", "") for n in notes if not n["continuation"])
        for _, notes in groupby(record["notes"], key=lambda n: (n["line_id"], n["word_id"]))
    )


def meter_entries(record):
    """Use explicit signatures and the same bar-grid origin as MIDI inference."""
    ppq = record["ticks_per_quarter"]
    if isinstance(ppq, bool) or not isinstance(ppq, int) or ppq < 1:
        raise AnnotationError("Invalid MIDI ticks_per_quarter")
    signatures = {}
    for event in record["time_signatures"]:
        tick = event["tick_960"]  # Calibrated exports use 960 PPQ, including fractional events.
        n, d = event["numerator"], event["denominator"]
        if (
            any(type(v) is not int or v < 1 for v in (n, d))
            or d & (d - 1)
            or not math.isfinite(tick)
            or tick < 0
        ):
            raise AnnotationError("Invalid explicit MIDI meter")
        if tick in signatures and signatures[tick] != (n, d):
            raise AnnotationError("Conflicting explicit MIDI meter")
        signatures[tick] = (n, d)
    ticks = sorted(signatures)
    result = []
    for note in record["notes"]:
        index = bisect_right(ticks, note["onset_tick"]) - 1
        if index < 0:
            result.append(None)
        else:
            origin = ticks[index]
            n, d = signatures[origin]
            beat = ((note["onset_tick"] - origin) * d / (4 * ppq)) % n
            result.append(dict(numerator=n, denominator=d, beat_position=beat))
    return result


def extract_song(record, *, max_syllables=40, require_meter=True):
    """Keep processed notes unchanged; adapt their word groups to shared IPA extraction."""
    if record["orthographic_alignment"] != "count_consistent_index_mapping":
        raise AnnotationError("Spelling alignment is unresolved")
    if not record["deduplication"]["is_representative"]:
        raise AnnotationError("Use the deduplicated orthographic export")
    if record["ticks_per_quarter"] != 960 or not record.get("timing_verification"):
        raise AnnotationError("Use calibrated 960-PPQ alignments")
    if record["seconds_status"] == "unresolved_tempo_metadata":
        raise AnnotationError("Unresolved tempo metadata")
    notes = record["notes"]
    if not notes:
        raise AnnotationError("Empty melody")
    meter = meter_entries(record)
    if require_meter and any(entry is None for entry in meter):
        raise AnnotationError("Missing explicit meter at a melody note")
    annot = dict(lines=[], words=[], notes=[])
    previous_end, previous_tick = -1.0, -1
    for note in notes:
        start, end = note["onset_seconds"], note["offset_seconds"]
        if (
            start is None
            or end is None
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end
        ):
            raise AnnotationError("Missing or invalid calibrated seconds")
        if (
            type(note["pitch"]) is not int
            or not 0 <= note["pitch"] <= 127
            or type(note["onset_tick"]) is not int
            or type(note["offset_tick"]) is not int
            or note["onset_tick"] < 0
            or note["onset_tick"] < previous_tick
            or note["offset_tick"] <= note["onset_tick"]
            or note["duration_ticks"] != note["offset_tick"] - note["onset_tick"]
            or start < previous_end - 1e-6
        ):
            raise AnnotationError("Invalid or overlapping melody notes")
        previous_end, previous_tick = end, note["offset_tick"]
    source_lines = []
    seen_lines = set()
    for source_line_id, indexed in groupby(enumerate(notes), key=lambda x: x[1]["line_id"]):
        if source_line_id in seen_lines:
            raise AnnotationError("Nonconsecutive source line membership")
        seen_lines.add(source_line_id)
        indexed = list(indexed)
        line_notes = [n for _, n in indexed]
        line_id = len(annot["lines"])
        source_lines.append(indexed)
        annot["lines"].append(
            dict(
                time=[line_notes[0]["onset_seconds"], line_notes[-1]["offset_seconds"]],
                index=line_notes[0]["paragraph_id"],
            )
        )
        seen_words = set()
        for word_id, group in groupby(line_notes, key=lambda n: n["word_id"]):
            if word_id in seen_words:
                raise AnnotationError("Nonconsecutive source word membership")
            seen_words.add(word_id)
            group = list(group)
            word = len(annot["words"])
            text = "".join(n["syllable_text"] for n in group if not n["continuation"])
            annot["words"].append(
                dict(
                    text=text,
                    index=line_id,
                    time=[group[0]["onset_seconds"], group[-1]["offset_seconds"]],
                )
            )
            annot["notes"].extend(
                dict(
                    text="~" if n["continuation"] else n["syllable_text"],
                    time=[n["onset_seconds"], n["offset_seconds"]],
                    index=word,
                )
                for n in group
            )
    song_id = f"lakh:{record['midi_md5']}"
    lines, rejected = extract_lines(
        {"id": song_id},
        annot,
        max_syllables=max_syllables,
        stress_source="ipa",
        prosody_mode="actual",
    )
    if rejected:
        raise AnnotationError("Unusable lyric line: " + rejected[0]["reason"])
    for line, indexed in zip(lines, source_lines, strict=True):
        ns = [n for _, n in indexed]
        line["source_line_id"] = ns[0]["line_id"]
        line["paragraph_id"] = ns[0]["paragraph_id"]
        line["note_alignment"]["scheme"] = LAKH_ALIGNMENT_SCHEME
        line["melody"] = dict(
            midi_pitches=[n["pitch"] for n in ns],
            onset_seconds=[n["onset_seconds"] for n in ns],
            note_duration_seconds=[n["offset_seconds"] - n["onset_seconds"] for n in ns],
            onset_ticks=[n["onset_tick"] for n in ns],
            note_duration_ticks=[n["duration_ticks"] for n in ns],
            source_note_indices=[i for i, _ in indexed],
            meter=[meter[i] for i, _ in indexed],
        )
    return dict(
        id=song_id,
        song_id=song_id,
        title="",
        lines=lines,
        source={
            "dataset": DATASET,
            **{
                key: record[key]
                for key in (
                    "source_file",
                    "midi_md5",
                    "msd_id",
                    "deduplication",
                    "ticks_per_quarter",
                    "original_ticks_per_quarter",
                    "tempo_map",
                    "time_signatures",
                    "seconds_status",
                    "timing_verification",
                )
            },
            "meter_phase": METER_PHASE,
            "note_policy": "authors_processed_notes_unchanged",
        },
    )


def prepare_lakh(config, *, limit=None, check_splits=False):
    data, model = config["data"], config.get("model", {})
    if (
        data.get("stress_source") != "ipa"
        or data.get("prosody_mode") != "actual"
        or data.get("include_melody") is not True
    ):
        raise ValueError(
            "Lakh requires stress_source: ipa, prosody_mode: actual, include_melody: true"
        )
    valid, test = data["valid_fraction"], data["test_fraction"]
    if not (0 < valid < 1 and 0 <= test < 1 and valid + test < 1):
        raise ValueError("Require valid_fraction > 0, test_fraction >= 0, and their sum < 1")
    if data["max_syllables"] < 1 or (limit is not None and limit < 1):
        raise ValueError("max_syllables and limit must be positive")
    if not isinstance(data.get("require_meter", True), bool):
        raise ValueError("require_meter must be boolean")
    alignment_policy = model.get("alignment_policy", "strict")
    if alignment_policy not in ALIGNMENT_POLICIES:
        raise ValueError("alignment_policy must be strict or prefix")
    limits = {
        k: model.get(k, default)
        for k, default in (
            ("max_song_notes", 2048),
            ("max_song_lines", 256),
            ("max_target_length", 4096),
        )
    }
    if any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError("Song and target limits must be positive integers")
    if data.get("pretraining_manifest"):
        raise ValueError("Lakh preparation does not use a DALI contrastive pretraining manifest")
    source, output = Path(data["lakh_file"]), Path(data["prepared_dir"])
    source_hash = file_sha256(source)
    requested, selection_hash = None, None
    if data.get("song_ids_file"):
        path = Path(data["song_ids_file"])
        requested = {
            s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()
        }
        if not requested:
            raise ValueError("song_ids_file must contain exact lakh:<midi_md5> IDs")
        selection_hash = file_sha256(path)
    candidates, seen_lyrics = {}, {}
    for record in read_records(source, limit):
        song_id = f"lakh:{record['midi_md5']}"
        if requested is not None and song_id not in requested:
            continue
        if song_id in candidates:
            raise ValueError("Duplicate MIDI hash; use the deduplicated orthographic export")
        identity = normalized_identity(lyric_text(record))
        group = seen_lyrics.setdefault(identity, song_id) if len(identity) >= 100 else song_id
        candidates[song_id] = dict(
            group=group,
            split=split_for(group, data["seed"], valid, test),
            source_file=record["source_file"],
            midi_md5=record["midi_md5"],
        )
    if not candidates:
        raise ValueError("No candidate Lakh songs")
    output.mkdir(parents=True, exist_ok=True)
    if check_splits:
        report = dict(
            status="passed",
            candidate_songs=len(candidates),
            songs=candidates,
            split_counts=dict(Counter(s["split"] for s in candidates.values())),
        )
        (output / "split_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    backend()
    pronunciation = dict(
        package="prosodic",
        version=version("prosodic"),
        wordform="historical_dictionary_first_with_modern_fallback",
        rules=REPO_RULES,
        pronunciation_policy=HISTORICAL_POLICY,
        historical_dictionary=historical_provenance(),
    )
    counts, alignment_counts, rejections, songs = Counter(), Counter(), Counter(), {}
    with TemporaryDirectory(prefix=".prepare-", dir=output) as staging, ExitStack() as stack:
        staging = Path(staging)
        names = ("train.jsonl", "valid.jsonl", "test.jsonl", "rejected.jsonl")
        handles = {
            name: stack.enter_context((staging / name).open("w", encoding="utf-8"))
            for name in names
        }
        for record in tqdm(
            read_records(source, limit), total=len(candidates), desc="Preparing Lakh"
        ):
            song_id = f"lakh:{record['midi_md5']}"
            if song_id not in candidates:
                continue
            counts["songs_scanned"] += 1
            try:
                if len(record["notes"]) > limits["max_song_notes"]:
                    raise AnnotationError("Song exceeds max_song_notes")
                if len({n["line_id"] for n in record["notes"]}) > limits["max_song_lines"]:
                    raise AnnotationError("Song exceeds max_song_lines")
                song = extract_song(
                    record,
                    max_syllables=data["max_syllables"],
                    require_meter=data.get("require_meter", True),
                )
                if (
                    sum(len(line["syllables"]) + 1 for line in song["lines"])
                    > limits["max_target_length"]
                ):
                    raise AnnotationError("Song exceeds max_target_length")
                _, aligned = aligned_note_indices(song["lines"], alignment_policy)
                if not aligned["supervised_slots"]:
                    raise AnnotationError("Song has no aligned IPA targets")
            except (AnnotationError, KeyError, TypeError, ValueError) as exc:
                counts["songs_rejected"] += 1
                rejections[str(exc)] += 1
                handles["rejected.jsonl"].write(
                    json.dumps(
                        dict(
                            song_id=song_id,
                            source_file=record["source_file"],
                            reason=str(exc),
                        )
                    )
                    + "\n"
                )
                continue
            info = candidates[song_id]
            split = info["split"]
            songs[song_id] = info
            handles[f"{split}.jsonl"].write(json.dumps(song, ensure_ascii=False) + "\n")
            counts[f"{split}_songs"] += 1
            counts[f"{split}_lines"] += len(song["lines"])
            counts["notes"] += len(record["notes"])
            counts["notes_with_meter"] += sum(
                entry is not None for line in song["lines"] for entry in line["melody"]["meter"]
            )
            counts["ipa_syllables"] += sum(len(line["syllables"]) for line in song["lines"])
            alignment_counts.update(aligned)
        if not songs:
            raise ValueError("No usable Lakh songs: " + json.dumps(dict(rejections)))
        if file_sha256(source) != source_hash:
            raise ValueError("Lakh source changed during preparation")
        stack.close()
        manifest = dict(
            schema_version=SCHEMA_VERSION,
            dataset=DATASET,
            config=data,
            pronunciation=pronunciation,
            partial=limit is not None,
            songs=songs,
            counts=dict(counts),
            limits=limits,
            source=dict(path=str(source.resolve()), sha256=source_hash),
            alignment_scheme=LAKH_ALIGNMENT_SCHEME,
            alignment_policy=alignment_policy,
            meter_phase=METER_PHASE,
            alignment_counts=dict(alignment_counts),
            rejection_reasons=dict(rejections),
            grouping_policy="midi_hash_and_exact_full_lyrics_at_least_100_characters_v1",
            selection=dict(
                sha256=selection_hash,
                requested_ids=sorted(requested) if requested is not None else None,
                missing_ids=sorted(requested - candidates.keys()) if requested else [],
                unprepared_ids=sorted(requested - songs.keys()) if requested else [],
            ),
            sha256={name: file_sha256(staging / name) for name in names},
        )
        for name in names:
            (staging / name).replace(output / name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "manifest.json").replace(output / "manifest.json")
    return manifest
