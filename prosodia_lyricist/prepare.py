"""Prepare reproducible song-disjoint DALI JSONL splits without audio or MIDI."""

import argparse
import hashlib
import json
import math
import pickle
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm

from .config import load_config
from .dali import AnnotationError, extract_lines, normalize_text, read_annotation

SCHEMA_VERSION = 2


def normalized_identity(value):
    return re.sub(r"\W+", "", normalize_text(value).casefold())


def song_groups(songs):
    """Keep duplicate artist/title entries and repeated audio ids in one split."""
    parent = {song_id: song_id for song_id in songs}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for song_id, info in sorted(songs.items()):
        artist = normalized_identity(info.get("artist", ""))
        title = normalized_identity(info.get("title", ""))
        audio = info.get("audio", {}).get("url", "")
        identities = []
        if artist not in ("", "none") and title not in ("", "none"):
            identities.append(("song", artist, title))
        if audio and audio != "None":
            identities.append(("audio", audio))
        for identity in identities:
            if identity in seen:
                a, b = sorted((root(song_id), root(seen[identity])))
                parent[b] = a
            else:
                seen[identity] = song_id
    return {song_id: root(song_id) for song_id in songs}


def split_for(group, seed, valid_fraction, test_fraction):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + valid_fraction:
        return "valid"
    return "train"


def prepare(config, *, limit=None):
    data = config["data"]
    valid, test = data["valid_fraction"], data["test_fraction"]
    if not (0 < valid < 1 and 0 <= test < 1 and valid + test < 1):
        raise ValueError("Require valid_fraction > 0, test_fraction >= 0, and their sum < 1")
    if data["max_syllables"] < 1 or (limit is not None and limit < 1):
        raise ValueError("max_syllables and limit must be positive")
    if data["stress_source"] not in ("ipa", "lexical", "unknown"):
        raise ValueError("stress_source must be ipa, lexical or unknown")
    if data["stress_source"] == "ipa":
        from .ipa import backend

        backend()  # Fail before creating outputs if pronunciation support is unavailable.
    requested, selection_hash = None, None
    if data.get("song_ids_file"):
        selection = Path(data["song_ids_file"]).read_bytes()
        requested = set(selection.decode("utf-8-sig").splitlines())
        requested = {s.strip() for s in requested if s.strip()}
        if not requested or any(len(s.split()) != 1 for s in requested):
            raise ValueError("song_ids_file must contain one exact song ID per nonempty line")
        selection_hash = hashlib.sha256(selection).hexdigest()
    seen_ids = set()
    source = Path(data["dali_dir"])
    files = sorted(p for p in source.iterdir() if p.suffix in (".gz", ".json"))
    if not files:
        raise ValueError(f"No DALI .gz or .json files in {source}")
    files = files[:limit] if limit else files
    output = Path(data["prepared_dir"])
    output.mkdir(parents=True, exist_ok=True)
    songs, counts = {}, Counter()
    with TemporaryDirectory(prefix=".prepare-", dir=output) as staging, ExitStack() as stack:
        staging = Path(staging)
        rejects = stack.enter_context((staging / "rejected.jsonl").open("w", encoding="utf-8"))
        rows = stack.enter_context((staging / "rows.jsonl").open("w+", encoding="utf-8"))
        for path in tqdm(files, desc="Preparing DALI", unit="song"):
            counts["files_scanned"] += 1
            try:
                info, annot = read_annotation(path)
                song_id = str(info["id"])
                if requested is not None and song_id not in requested:
                    counts["songs_filtered_id"] += 1
                    continue
                seen_ids.add(song_id)
                if song_id in songs:
                    raise AnnotationError("Duplicate song id (keep only one export per song)")
                language = str(info.get("metadata", {}).get("language", "")).lower()
                if data["language"] != "all" and language != data["language"].lower():
                    counts["songs_filtered_language"] += 1
                    continue
                ncc = float(info.get("scores", {}).get("NCC", 0.0))
                if not math.isfinite(ncc) or ncc < data["min_ncc"]:
                    counts["songs_filtered_ncc"] += 1
                    continue
                records, rejected = extract_lines(
                    info,
                    annot,
                    max_syllables=data["max_syllables"],
                    stress_source=data["stress_source"],
                )
                for rejection in rejected:
                    rejects.write(json.dumps(rejection) + "\n")
                counts["lines_rejected"] += len(rejected)
                if not records:
                    counts["songs_without_usable_lines"] += 1
                    continue
                songs[song_id] = info
                for record in records:
                    rows.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["syllables"] += len(record["syllables"])
                    counts["unknown_stress_syllables"] += sum(
                        s["stress"] == "unknown" for s in record["syllables"]
                    )
            except (
                AnnotationError,
                OSError,
                EOFError,
                pickle.UnpicklingError,
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
            ) as exc:
                counts["files_rejected"] += 1
                rejects.write(json.dumps({"file": path.name, "reason": str(exc)}) + "\n")
        if not songs:
            raise ValueError("No usable DALI songs; check language, quality, and annotation format")
        groups = song_groups(songs)
        splits = {
            song_id: split_for(group, data["seed"], valid, test)
            for song_id, group in groups.items()
        }
        handles = {
            split: stack.enter_context((staging / f"{split}.jsonl").open("w", encoding="utf-8"))
            for split in ("train", "valid", "test")
        }
        rows.seek(0)
        for row in rows:
            record = json.loads(row)
            split = splits[record["song_id"]]
            handles[split].write(row)
            counts[f"{split}_lines"] += 1
        for split in splits.values():
            counts[f"{split}_songs"] += 1
        # Close files before publishing; manifest is written last as the completion marker.
        stack.close()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": data,
            "partial": limit is not None,
            "selection": {
                "sha256": selection_hash,
                "requested_ids": sorted(requested) if requested is not None else None,
                "missing_ids": sorted(requested - seen_ids) if requested is not None else [],
                "unprepared_ids": sorted(requested - songs.keys()) if requested is not None else [],
            },
            "counts": dict(counts),
            "songs": {
                song_id: {"split": splits[song_id], "group": groups[song_id]}
                for song_id in sorted(songs)
            },
            "sha256": {},
        }
        for name in ("train.jsonl", "valid.jsonl", "test.jsonl", "rejected.jsonl"):
            path = staging / name
            manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            path.replace(output / name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging / "manifest.json").replace(output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="Scan only the first N files for a smoke test")
    parser.add_argument("--song-ids-file", help="Text file containing one exact DALI ID per line")
    parser.add_argument(
        "--english-only", action="store_true", help="Exclude other/unknown languages"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.song_ids_file:
        config["data"]["song_ids_file"] = str(Path(args.song_ids_file).expanduser().resolve())
    if args.english_only:
        config["data"]["language"] = "english"
    manifest = prepare(config, limit=args.limit)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
