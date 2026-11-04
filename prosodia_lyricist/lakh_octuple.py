"""Migrate verified Lakh IPA data to canonical timing without changing target labels."""

import copy
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm

from .bridge_alignment import aligned_note_indices
from .data import read_manifest
from .melody_checkpoint import file_sha256
from .melody_encoder.octuple import GRID_POLICY, canonical_meter, codec_metadata, encode_octuple
from .prepare import song_groups
from .split_integrity import (
    content_fingerprints,
    read_content_index,
    require_clean_splits,
    source_msd_ids,
)


def prepare_canonical_lakh(config, *, limit=None, check_splits=False):
    from .lakh import read_records

    data, model = config["data"], config["model"]
    base, output = Path(data["base_prepared_dir"]).resolve(), Path(data["prepared_dir"]).resolve()
    if base == output or base in output.parents or output in base.parents:
        raise ValueError("Canonical preparation requires a separate output directory")
    if data.get("source") != "lakh" or not data.get("require_meter", True):
        raise ValueError("Canonical preparation requires Lakh with explicit meter")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    manifest = read_manifest(base)
    for key in (
        "source",
        "stress_source",
        "prosody_mode",
        "include_melody",
        "max_syllables",
        "seed",
        "valid_fraction",
        "test_fraction",
        "song_ids_file",
    ):
        if data.get(key) != manifest["config"].get(key):
            raise ValueError(
                f"Base Lakh preparation differs in {key}; retain its target/split policy"
            )
    if manifest.get("partial"):
        raise ValueError("Use a complete base prepared dataset")
    source_hash = file_sha256(data["lakh_file"])
    if source_hash != manifest.get("source", {}).get("sha256"):
        raise ValueError("Raw Lakh export differs from the base preparation")
    base_hash = file_sha256(base / "manifest.json")
    require_clean_splits(read_content_index(base))
    if check_splits:
        return dict(
            status="passed",
            basis="verified_base_splits",
            candidate_songs=len(manifest["songs"]),
            base_manifest_sha256=base_hash,
            split_counts=dict(Counter(info["split"] for info in manifest["songs"].values())),
        )
    # Only source note identities and role targets are needed; do not re-pronounce lyrics.
    raw_notes = {}
    for raw in read_records(data["lakh_file"]):
        sid = "lakh:" + raw["midi_md5"]
        if sid in manifest["songs"]:
            if sid in raw_notes:
                raise ValueError("Duplicate raw MIDI identity")
            raw_notes[sid] = [
                (n["pitch"], n["onset_tick"], n["duration_ticks"], n["continuation"])
                for n in raw["notes"]
            ]
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".canonical-", dir=output) as temporary:
        staging = Path(temporary)
        songs, counts, diagnostics, roles, alignment = (
            {},
            Counter(),
            Counter(),
            Counter(),
            Counter(),
        )
        with (staging / "rows.jsonl").open("w") as rows:
            for split in ("train", "valid", "test"):
                with (base / f"{split}.jsonl").open() as handle:
                    selected = 0
                    for text in tqdm(handle, desc=f"Canonical Lakh {split}", unit="song"):
                        if limit is not None and selected >= limit:
                            break
                        song = json.loads(text)
                        sid = song["song_id"]
                        notes = raw_notes.get(sid)
                        if notes is None:
                            raise ValueError(f"Missing raw source for {sid}")
                        offset = 0
                        for line in song["lines"]:
                            m = line["melody"]
                            identity = list(
                                zip(
                                    m["midi_pitches"],
                                    m["onset_ticks"],
                                    m["note_duration_ticks"],
                                    strict=True,
                                )
                            )
                            original = notes[offset : offset + len(identity)]
                            if identity != [n[:3] for n in original]:
                                raise ValueError(f"Raw/prepared note identity mismatch: {sid}")
                            if any(type(n[3]) is not bool for n in original):
                                raise ValueError("Continuation annotation must be boolean")
                            line["note_roles"] = [int(n[3]) for n in original]
                            if line["note_roles"][0]:
                                raise ValueError("Orphan continuation at supplied phrase start")
                            roles.update(f"{split}_{r}" for r in line["note_roles"])
                            offset += len(identity)
                        if offset != len(notes):
                            raise ValueError("Prepared song omits raw notes")
                        meter, diag = canonical_meter(song["lines"], song["source"])
                        offset = 0
                        for line in song["lines"]:
                            length = len(line["note_roles"])
                            line["melody"]["meter"] = meter[offset : offset + length]
                            offset += length
                        song["source"]["meter_phase"] = GRID_POLICY
                        diagnostics.update(diag)
                        encode_octuple(song["lines"], song["source"], model.get("octuple_config"))
                        _, aligned = aligned_note_indices(
                            song["lines"], model.get("alignment_policy", "strict")
                        )
                        alignment.update(aligned)
                        songs[sid] = {
                            **manifest["songs"][sid],
                            "msd_ids": source_msd_ids(song["source"]),
                            "content_fingerprints": content_fingerprints(
                                song["lines"],
                                song["source"],
                                octuple_config=model.get("octuple_config"),
                            ),
                        }
                        counts["notes"] += len(notes)
                        counts["ipa_syllables"] += sum(
                            len(line["syllables"]) for line in song["lines"]
                        )
                        rows.write(json.dumps(song, ensure_ascii=False) + "\n")
                        selected += 1
        # Union new identities with existing groups; preserve membership unless a collision
        # crosses splits, in which case move the entire group toward train, then valid.
        groups = song_groups(songs)
        rank = {"train": 0, "valid": 1, "test": 2}
        split_by_group = {}
        for sid, info in songs.items():
            group = groups[sid]
            split_by_group[group] = min(
                split_by_group.get(group, info["split"]), info["split"], key=rank.get
            )
        changes = []
        for sid, info in songs.items():
            split = split_by_group[groups[sid]]
            if info["split"] != split:
                changes.append(dict(song_id=sid, previous=info["split"], split=split))
            info.update(group=groups[sid], split=split)
        # Role weights must reflect the final train membership, including any moved groups.
        roles.clear()
        from contextlib import ExitStack

        with ExitStack() as stack:
            handles = {
                s: stack.enter_context((staging / f"{s}.jsonl").open("w"))
                for s in ("train", "valid", "test")
            }
            with (staging / "rows.jsonl").open() as rows:
                for text in rows:
                    song = json.loads(text)
                    split = songs[song["song_id"]]["split"]
                    handles[split].write(text)
                    counts[f"{split}_songs"] += 1
                    counts[f"{split}_lines"] += len(song["lines"])
                    roles.update(
                        f"{split}_{r}" for line in song["lines"] for r in line["note_roles"]
                    )
        if (
            file_sha256(data["lakh_file"]) != source_hash
            or file_sha256(base / "manifest.json") != base_hash
        ):
            raise ValueError("Lakh source/base changed during preparation")
        result = copy.deepcopy(manifest)
        result.update(
            config=data,
            songs=songs,
            counts=dict(counts),
            partial=limit is not None,
            alignment_counts=dict(alignment),
            alignment_policy=model.get("alignment_policy", "strict"),
            meter_phase=GRID_POLICY,
            octuple_codec=codec_metadata(model.get("octuple_config")),
            grid_diagnostics=dict(diagnostics),
            note_role_counts=dict(roles),
            preparation_parent=dict(path=str(base), sha256=base_hash, split_changes=changes),
            source=dict(path=str(Path(data["lakh_file"]).resolve()), sha256=source_hash),
        )
        (staging / "rejected.jsonl").write_text("")
        names = ["train.jsonl", "valid.jsonl", "test.jsonl", "rejected.jsonl"]
        result["sha256"] = {name: file_sha256(staging / name) for name in names}
        (staging / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        require_clean_splits(read_content_index(staging))
        for name in [*names, "manifest.json"]:
            (staging / name).replace(output / name)
    return result
