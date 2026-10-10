"""Export DALI song IDs from prepared bridge validation and test splits."""

import argparse
import hashlib
import json
from pathlib import Path


def export_ids(prepared_dir, output_dir):
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    split_ids = {}
    for split in ("valid", "test"):
        name = f"{split}.jsonl"
        contents = (prepared_dir / name).read_bytes()
        if hashlib.sha256(contents).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"{name} differs from the preparation manifest")
        ids = [json.loads(row)["song_id"] for row in contents.splitlines() if row.strip()]
        expected = {
            song_id for song_id, info in manifest["songs"].items() if info["split"] == split
        }
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(f"{name} song IDs differ from the preparation manifest")
        split_ids[split] = sorted(ids)
    if set(split_ids["valid"]) & set(split_ids["test"]):
        raise ValueError("Validation and test songs overlap")
    split_ids["heldout"] = sorted(split_ids["valid"] + split_ids["test"])

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in split_ids.items():
        path = output_dir / f"{split}_dali_ids.txt"
        path.write_text("".join(f"{song_id}\n" for song_id in ids), encoding="utf-8")
        print(f"{split}: {len(ids)} songs -> {path}")
    return split_ids


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=root / "data/dali-bridge")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/bridge-split-ids")
    args = parser.parse_args()
    export_ids(args.prepared_dir, args.output_dir)


if __name__ == "__main__":
    main()
