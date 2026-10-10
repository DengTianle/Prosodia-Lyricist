"""Audit prepared bridge splits against a contrastive CSV without extracting notes.

Run with python -m prosodia_lyricist.audit_bridge_splits --help.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from .dali import read_annotation
from .data import read_manifest
from .melody_checkpoint import file_sha256
from .melody_data import read_melody_manifest
from .prepare import normalized_identity, song_groups

logger = logging.getLogger(__name__)


def _has_identity(info):
    return bool(
        info.get("audio", {}).get("url") not in (None, "", "None")
        or all(normalized_identity(info.get(key, "")) not in ("", "none")
               for key in ("artist", "title"))
    )


def audit_prepared_splits(prepared_dir, pretraining_manifest, *, dali_dir=None):
    """Return a report; missing/invalid upstream CSVs are explicitly unavailable.

    Prepared-data integrity errors always raise. Legacy identity recovery only
    reads DALI metadata; it never calls preparation, IPA, or note extraction.
    """
    prepared_dir = Path(prepared_dir)
    manifest = read_manifest(prepared_dir)
    for name, expected in manifest["sha256"].items():
        if name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            if file_sha256(prepared_dir / name) != expected:
                raise ValueError(f"{name} differs from its preparation manifest")
    report = {
        "audit_version": 1,
        "status": "unavailable",
        "prepared_manifest_sha256": file_sha256(prepared_dir / "manifest.json"),
        "data_sha256": manifest["sha256"],
        "manifest": str(Path(pretraining_manifest).resolve()) if pretraining_manifest else None,
        "sha256": None,
        "line_counts": [],
        "conflicts": [],
        "limitations": [
            "Checks supplied metadata only; the checkpoint does not record a historical CSV hash.",
            "Audio identities use exact paths/URLs, not audio-content fingerprints.",
        ],
    }
    if not pretraining_manifest:
        report["reason"] = "No contrastive pretraining manifest supplied"
        return report
    try:
        report["sha256"] = file_sha256(pretraining_manifest)
        rows, upstream = read_melody_manifest(pretraining_manifest, require_lyrics=False)
        report["line_counts"] = sorted({int(row["line_count"]) for row in rows
                                        if row.get("line_count")})
    except (OSError, ValueError) as exc:
        report["reason"] = f"Cannot audit contrastive manifest: {exc}"
        return report

    downstream = {song: info.get("identity", {}) for song, info in manifest["songs"].items()}
    missing = {song for song, info in downstream.items() if not _has_identity(info)}
    if dali_dir and missing:
        # Filenames are not necessarily song IDs in exported DALI JSON datasets.
        for path in sorted(Path(dali_dir).iterdir()):
            if path.suffix not in (".gz", ".json"):
                continue
            info, _ = read_annotation(path)
            song = str(info["id"])
            if song in missing:
                downstream[song] = info
                missing.remove(song)
            if not missing:
                break

    combined = {f"up:{song}": info for song, info in upstream.items()}
    combined.update({f"down:{song}": info for song, info in downstream.items()})
    parents = song_groups(combined)

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(a, b):
        parents[root(a)] = root(b)

    # Preserve transitive identity links through both exact IDs and known groups.
    shared = upstream.keys() & downstream.keys()
    for song in shared:
        join(f"up:{song}", f"down:{song}")
    prepared_groups = {}
    for song, info in manifest["songs"].items():
        key = f"down:{song}"
        join(key, prepared_groups.setdefault(info["group"], key))
    groups = defaultdict(lambda: {"upstream": [], "downstream": []})
    for side, songs in (("upstream", upstream), ("downstream", manifest["songs"])):
        prefix = "up" if side == "upstream" else "down"
        for song, info in songs.items():
            groups[root(f"{prefix}:{song}")][side].append({"song_id": song, "split": info["split"]})
    for group in groups.values():
        members = group["upstream"] + group["downstream"]
        if not group["downstream"] or len({member["split"] for member in members}) < 2:
            continue
        train_exposure = any(row["split"] == "train" for row in group["upstream"]) and any(
            row["split"] in ("valid", "test") for row in group["downstream"]
        )
        report["conflicts"].append({
            "kind": "pretraining_train_in_downstream_heldout" if train_exposure
                    else "split_mismatch",
            **group,
        })
    report["shared_songs"] = len(shared)
    report["downstream_only_songs"] = len(downstream.keys() - upstream.keys())
    report["missing_downstream_identities"] = sorted(
        song for song, info in downstream.items() if not _has_identity(info)
    )
    report["missing_upstream_identities"] = sorted(
        song for song, info in upstream.items() if not _has_identity(info)
    )
    original_audit = manifest.get("pretraining_split_audit") or {}
    report["matches_preparation_audit"] = original_audit.get("sha256") == report["sha256"]
    if report["conflicts"]:
        report["status"] = "failed"
        report["reason"] = f"{len(report['conflicts'])} identity groups cross splits"
    elif report["matches_preparation_audit"]:
        report["status"] = "passed"
        report["basis"] = "unchanged_preparation_audit"
    elif report["missing_downstream_identities"] or report["missing_upstream_identities"]:
        report["status"] = "incomplete"
        report["reason"] = (
            "Song-ID and known-group checks completed, but duplicate identity metadata is missing. "
            "For legacy prepared data, supply --dali-dir to read original song identities."
        )
    else:
        report["status"] = "passed"
        report["basis"] = "song_ids_and_duplicate_identities"
    return report


def check_training_splits(data, manifest, *, mode, required, encoder_lines):
    """Keep legacy strict enforcement, or audit the current CSV and warn."""
    if mode not in ("strict", "warning"):
        raise ValueError("training.pretraining_audit_mode must be strict or warning")
    path = data.get("pretraining_manifest")
    if not path and not required:
        return {"mode": mode, "status": "not_applicable"}
    if mode == "strict":
        previous = manifest.get("pretraining_split_audit")
        if not path or not previous or previous["sha256"] != file_sha256(path):
            raise ValueError(
                "Prepare with the unchanged melody pretraining manifest, or explicitly use "
                "training.pretraining_audit_mode: warning to audit without re-preparation"
            )
    report = audit_prepared_splits(data["prepared_dir"], path)
    report["mode"] = mode
    sizes = set(report["line_counts"])
    if sizes and sizes != {encoder_lines}:
        raise ValueError("lines_per_window must match the contrastive pretraining manifest")
    if report["status"] != "passed":
        message = f"Pretraining split audit {report['status']}: {report['reason']}"
        if mode == "strict":
            raise ValueError(message)
        logger.warning("%s; continuing in warning mode. Held-out cleanliness is not verified.",
                       message)
    else:
        logger.info("Pretraining split audit passed (%s)", report["basis"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--pretraining-manifest", required=True, type=Path)
    parser.add_argument("--dali-dir", type=Path,
                        help="Optional original DALI directory for legacy identity metadata")
    parser.add_argument("--output", type=Path, default=Path("outputs/bridge-split-audit.json"))
    args = parser.parse_args()
    report = audit_prepared_splits(args.prepared_dir, args.pretraining_manifest,
                                   dali_dir=args.dali_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Split audit: {report['status']}; {len(report['conflicts'])} conflicting groups")
    if report.get("reason"):
        print(report["reason"])
    print(f"Report: {args.output}")
    exit_codes = {"passed": 0, "failed": 1, "incomplete": 2, "unavailable": 2}
    raise SystemExit(exit_codes[report["status"]])


if __name__ == "__main__":
    main()
