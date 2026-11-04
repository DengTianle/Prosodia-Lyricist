"""Content identities, supervised exposure checks, and non-destructive split repair."""

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

CONTENT_POLICY = "metadata_encoded_melody_full_lyrics_v1"
GROUP_FINGERPRINTS = ("input", "melody", "lyrics", "canonical_melody", "octuple_input")


def _digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_digest(encoded, keys):
    digest = hashlib.sha256()
    for key in keys:
        array = np.asarray(encoded[key])
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        digest.update(json.dumps([key, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def content_fingerprints(lines, source=None, *, octuple_config=None):
    """Hash what the encoder sees, separately from targets and full lyric text.

    The melody identity excludes optional meter so disabling that branch cannot
    hide a duplicate. Short/generic lyrics alone do not establish song identity.
    Target equality is diagnostic only and never joins identity groups.
    """
    result = {}
    text = unicodedata.normalize("NFKC", " ".join(line.get("text", "") for line in lines))
    words = " ".join(re.findall(r"\w+(?:['’]\w+)*", text.casefold()))
    if len(re.sub(r"\W", "", words)) >= 100:
        result["lyrics"] = _digest(words)
    if lines and all("melody" in line for line in lines):
        from .bridge_data import encode_bridge_song

        encoded = encode_bridge_song(lines)
        keys = ("melody_features", "note_line_ids", "line_count")
        result["melody"] = _array_digest(encoded, keys)
        result["input"] = _array_digest(encoded, (*keys, "meter_features", "meter_mask"))
        if source and source.get("meter_phase") == "explicit_meter_continuous_bars_v1":
            from .melody_encoder.octuple import canonical_notes, encode_octuple

            notes, _, _ = canonical_notes(lines, source)
            origin = notes[0][1]
            # Exclude phrase boundaries, tempo and optional meter from melody identity.
            result["canonical_melody"] = _digest([
                (pitch, str(onset-origin), str(duration)) for pitch, onset, duration in notes
            ])
            octuple = encode_octuple(lines, source, octuple_config)
            if octuple_config and not octuple_config.get("timing_residuals", True):
                octuple["timing_residuals"] = np.zeros_like(octuple["timing_residuals"])
            result["octuple_input"] = _array_digest(octuple, (
                "melody_features", "timing_residuals", "note_line_ids", "line_count",
            ))
    if lines and all("syllables" in line for line in lines):
        result["target"] = _digest([
            [(s["stress"], s["length"]) for s in line["syllables"]] for line in lines
        ])
    return result


def read_content_index(directory, *, octuple_config=None):
    """Verify actual JSONL bytes and index all splits, including legacy exports."""
    from .data import read_manifest
    from .melody_checkpoint import file_sha256

    directory = Path(directory).resolve()
    manifest = read_manifest(directory)
    if octuple_config is None:
        octuple_config = manifest.get("octuple_codec", {}).get("config")
    scope = file_sha256(directory / "manifest.json")
    entries, seen = [], set()
    for split in ("train", "valid", "test"):
        name = f"{split}.jsonl"
        if name not in manifest["sha256"]:
            if any(info["split"] == split for info in manifest["songs"].values()):
                raise ValueError(f"Missing {name} checksum in {directory}")
            continue
        if file_sha256(directory / name) != manifest["sha256"][name]:
            raise ValueError(f"{directory / name} differs from its preparation manifest")
        digest = hashlib.sha256()
        with (directory / name).open("rb") as handle:
            for row in handle:
                digest.update(row)
                song = json.loads(row)
                song_id = song["song_id"]
                info = manifest["songs"].get(song_id)
                if song_id in seen or info is None or info["split"] != split:
                    raise ValueError(f"Duplicate, undeclared or incorrectly split song: {song_id}")
                seen.add(song_id)
                lines = song["lines"]
                entries.append({
                    "song_id": song_id, "source": manifest["config"].get("source", "dali"),
                    "split": split, "scope": scope, "group": info["group"],
                    "identity": {**info.get("identity", {}),
                                 "msd_ids": source_msd_ids(song.get("source", {}))},
                    "content_fingerprints": content_fingerprints(
                        lines, song.get("source"), octuple_config=octuple_config),
                    "lines": len(lines),
                    "slots": sum(len(line.get("syllables", [])) for line in lines),
                })
        if digest.hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"{directory / name} differs from its preparation manifest")
    if seen != set(manifest["songs"]):
        raise ValueError(f"Prepared rows do not cover the song manifest in {directory}")
    return {
        "version": 1, "policy": CONTENT_POLICY, "prepared_dir": str(directory),
        "manifest_sha256": scope, "data_sha256": manifest["sha256"], "entries": entries,
    }


def source_msd_ids(source):
    ids = {source.get("msd_id", "")}
    ids.update(Path(path).parent.name for path in source.get("deduplication", {}).get(
        "source_aliases", []))
    return sorted(identity for identity in ids if isinstance(identity, str)
                  and re.fullmatch(r"TR[A-Z0-9]{16}", identity))


def identity_components(entries):
    from .prepare import song_groups

    infos = {
        str(i): {
            **entry["identity"],
            "source_song_id": f"{entry['source']}:{entry['song_id']}",
            "group": f"{entry['scope']}:{entry['group']}",
            "content_fingerprints": entry["content_fingerprints"],
        }
        for i, entry in enumerate(entries)
    }
    components = defaultdict(list)
    for key, group in song_groups(infos).items():
        components[group].append(int(key))
    return list(components.values())


def audit_content_splits(index, parent_exposure=None):
    """Check internal splits and training/selection exposure of every ancestor."""
    if parent_exposure is not None and (
        parent_exposure.get("version") != 1
        or parent_exposure.get("policy") != CONTENT_POLICY
        or not isinstance(parent_exposure.get("entries"), list)
        or not parent_exposure["entries"]
    ):
        raise ValueError("Missing or unsupported supervised exposure index in parent checkpoint")
    current = index["entries"]
    parents = (parent_exposure or {}).get("entries", [])
    entries = [*current, *parents]
    conflicts = []
    for indices in identity_components(entries):
        downstream = [entries[i] for i in indices if i < len(current)]
        upstream = [entries[i] for i in indices if i >= len(current)]
        splits = {e["split"] for e in downstream}
        kinds = []
        if len(splits) > 1:
            kinds.append("duplicate_content_or_identity_crosses_splits")
        if splits & {"valid", "test"} and any(e["split"] == "train" for e in upstream):
            kinds.append("ancestor_training_in_heldout")
        if "test" in splits and any(e["split"] == "valid" for e in upstream):
            kinds.append("ancestor_model_selection_in_test")
        if not kinds:
            continue
        repeated = {}
        for key in GROUP_FINGERPRINTS:
            values = Counter(e["content_fingerprints"].get(key) for e in [*downstream, *upstream])
            repeated[key] = sorted(v for v, count in values.items() if v and count > 1)
        conflicts.append({
            "kinds": kinds, "fingerprint_matches": repeated,
            "identical_targets": len({e["content_fingerprints"].get("target")
                                      for e in [*downstream, *upstream]}) == 1,
            "current": downstream, "ancestors": upstream,
        })
    affected = {(e["scope"], e["song_id"]): e for group in conflicts
                for e in group["current"] if e["split"] != "train"}
    return {
        "version": 1, "policy": CONTENT_POLICY,
        "status": "failed" if conflicts else "passed",
        "conflicts": conflicts,
        "affected_heldout": dict(Counter(e["split"] for e in affected.values())),
        "ancestor_exposure": "verified" if parent_exposure is not None else "not_supplied",
        "limitations": [
            "Exact encoded inputs and normalized full lyrics; not recording-identity verification.",
            "Lyrics shorter than 100 alphanumeric characters do not join groups on text alone.",
            "Near duplicates, covers, partial lyric overlap and audio fingerprints need review.",
        ],
    }


def require_clean_splits(index, parent_exposure=None):
    report = audit_content_splits(index, parent_exposure)
    if report["status"] != "passed":
        raise ValueError(
            f"Supervised split leakage: {len(report['conflicts'])} identity groups cross splits "
            "or ancestor training. Run python -m prosodia_lyricist.split_integrity "
            "--prepared-dir DIR --output REPORT.json; repair splits and retrain from scratch."
        )
    return report


def training_exposure(index, parent_exposure=None):
    """Carry the full ancestry forward, without recursively duplicating metadata."""
    entries = (parent_exposure or {}).get("entries", []) + [
        e for e in index["entries"] if e["split"] in ("train", "valid")
    ]
    unique = {_digest(e): e for e in entries}
    return {"version": 1, "policy": CONTENT_POLICY, "entries": list(unique.values())}


def repair_splits(indices, destination):
    """Copy datasets, moving entire connected groups to train > valid > test.

    Existing training examples never become held out. This repairs prospective
    training splits; it cannot undo what an existing checkpoint has learned.
    """
    from .melody_checkpoint import file_sha256

    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Repair destination must be new: {destination}")
    names = [Path(index["prepared_dir"]).name for index in indices]
    if len(names) != len(set(names)):
        raise ValueError("Prepared directories must have distinct names for repair")
    combined = {"entries": [e for index in indices for e in index["entries"]]}
    assignments = {}
    priority = {"train": 0, "valid": 1, "test": 2}
    for members in identity_components(combined["entries"]):
        group = [combined["entries"][i] for i in members]
        split = min((e["split"] for e in group), key=priority.get)
        group_id = "content:" + _digest(sorted((e["source"], e["song_id"]) for e in group))
        for entry in group:
            assignments[entry["scope"], entry["song_id"]] = split, group_id
    repaired, moves = [], []
    for index, name in zip(indices, names, strict=True):
        source = Path(index["prepared_dir"])
        # Recheck source bytes before publishing a new completion manifest.
        if read_content_index(source) != index:
            raise ValueError(f"Source changed during repair: {source}")
        output = destination / name
        output.mkdir(parents=True)
        manifest = copy.deepcopy(json.loads((source / "manifest.json").read_text()))
        manifest["config"]["prepared_dir"] = str(output)
        counts = Counter()
        by_id = {e["song_id"]: e for e in index["entries"]}
        with ExitStack() as stack:
            handles = {s: stack.enter_context((output / f"{s}.jsonl").open("wb"))
                       for s in ("train", "valid", "test")}
            for old_split in handles:
                path = source / f"{old_split}.jsonl"
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    for row in handle:
                        song_id = json.loads(row)["song_id"]
                        entry = by_id[song_id]
                        split, group = assignments[entry["scope"], song_id]
                        handles[split].write(row)
                        manifest["songs"][song_id].update(
                            split=split, group=group,
                            content_fingerprints=entry["content_fingerprints"],
                        )
                        counts[f"{split}_songs"] += 1
                        counts[f"{split}_lines"] += entry["lines"]
                        if split != old_split:
                            moves.append({"dataset": name, "song_id": song_id,
                                          "from": old_split, "to": split})
        for split in ("train", "valid", "test"):
            for unit in ("songs", "lines"):
                manifest.setdefault("counts", {})[f"{split}_{unit}"] = counts[f"{split}_{unit}"]
        if (source / "rejected.jsonl").exists():
            (output / "rejected.jsonl").write_bytes((source / "rejected.jsonl").read_bytes())
        manifest["sha256"] = {p.name: file_sha256(p) for p in output.glob("*.jsonl")}
        manifest["content_split_policy"] = CONTENT_POLICY
        manifest["split_repair"] = {
            "source": str(source), "manifest_sha256": index["manifest_sha256"],
            "policy": "connected_groups_train_then_valid_then_test_v1",
        }
        # A prior metadata/pretraining audit cannot certify the new assignments.
        if manifest.get("pretraining_split_audit"):
            manifest["pretraining_split_audit"] = None
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        repaired.append(read_content_index(output))
    report = require_clean_splits({"entries": [e for i in repaired for e in i["entries"]]})
    return {"prepared_dirs": [i["prepared_dir"] for i in repaired],
            "reassigned_songs": moves, "audit": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", action="append", required=True, type=Path,
                        help="Repeat to audit or repair DALI and Lakh together")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path,
                        help="New directory for repaired copies; sources remain unchanged")
    args = parser.parse_args(argv)
    indices = [read_content_index(p) for p in args.prepared_dir]
    report = audit_content_splits({"entries": [e for i in indices for e in i["entries"]]})
    report["datasets"] = [{k: v for k, v in i.items() if k != "entries"} for i in indices]
    if args.repair_root:
        report["repair"] = repair_splits(indices, args.repair_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Content split audit: {report['status']}; {len(report['conflicts'])} conflict groups")
    if args.repair_root:
        print(f"Repaired copies: {', '.join(report['repair']['prepared_dirs'])}")
    print(f"Report: {args.output}")
    raise SystemExit(0 if args.repair_root or report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
