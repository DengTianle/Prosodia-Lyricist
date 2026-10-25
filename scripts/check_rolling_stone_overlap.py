"""Compare Rolling Stone .str lyrics with the downloaded Lyricist training CSV.

Title matches generate candidates; shared 14-character lyric sequences confirm
overlap. Covers count, as their lyrics can also expose evaluation text during
training. Scores measure coverage of unique subset lyric sequences, not the
fraction of words that are identical. Record numbers are one-based CSV data
records, excluding the header (lyrics contain newlines).
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def lyric_sequences(text):
    text = normalize(text)
    return {text[i : i + 14] for i in range(len(text) - 13)}


def compare(subset_dir, train_csv, output, min_coverage=0.25):
    references = {}
    for path in sorted(subset_dir.glob("*.str")):
        # Each syllable repeats its full word. Retain only syllable [1].
        words = re.findall(r"(\S+)\[1\]", path.read_text(encoding="utf-8"))
        sequences = lyric_sequences(" ".join(words))
        if not sequences:
            raise ValueError(f"No usable lyrics in {path}")
        references[path.stem] = sequences
    if not references:
        raise ValueError(f"No .str files in {subset_dir}")

    titles = {song: normalize(song) for song in references}
    matches = {song: [] for song in references}
    best_candidates = {song: None for song in references}
    csv.field_size_limit(10_000_000)
    record_count = 0
    with train_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Artist", "Song", "Lyrics"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected CSV columns: {sorted(required)}")
        for record_count, row in enumerate(reader, 1):
            title = normalize(row["Song"])
            candidates = [
                song
                for song, expected in titles.items()
                if title == expected or (len(expected) >= 8 and expected in title)
            ]
            if not candidates:
                continue
            sequences = lyric_sequences(row["Lyrics"])
            for song in candidates:
                shared = len(references[song] & sequences)
                coverage = shared / len(references[song])
                candidate = {
                    "record": record_count,
                    "artist": row["Artist"],
                    "title": row["Song"],
                    "lyric_sequence_coverage": round(coverage, 6),
                    "shared_sequences": shared,
                }
                best = best_candidates[song]
                if best is None or coverage > best["lyric_sequence_coverage"]:
                    best_candidates[song] = candidate
                if coverage >= min_coverage:
                    matches[song].append(candidate)

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subset_song", "status", "matching_train_records", "matched_artists",
        "best_title_candidate", "train_matches",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for song, song_matches in matches.items():
            song_matches.sort(key=lambda item: -item["lyric_sequence_coverage"])
            writer.writerow({
                "subset_song": song,
                "status": "overlap" if song_matches else "no_confirmed_overlap",
                "matching_train_records": len(song_matches),
                "matched_artists": "; ".join(sorted({m["artist"] for m in song_matches})),
                "best_title_candidate": json.dumps(best_candidates[song], ensure_ascii=False),
                "train_matches": json.dumps(song_matches, ensure_ascii=False),
            })

    summary = Counter("overlap" if rows else "no_confirmed_overlap" for rows in matches.values())
    print(f"Training records scanned: {record_count:,}")
    print(f"Subset songs: {len(references)}")
    print(f"Songs with lyric overlap (including covers): {summary['overlap']}")
    print(f"Songs without confirmed overlap: {summary['no_confirmed_overlap']}")
    print(f"Minimum lyric-sequence coverage: {min_coverage:.0%}")
    for song, rows in matches.items():
        if not rows:
            print(f"  No confirmed overlap: {song}")
    print(f"Report: {output}")
    return matches


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, default=root / "data/stress-format-space")
    parser.add_argument("--train-csv", type=Path, default=root / "data/lyricist-dataset/train.csv")
    parser.add_argument("--output", type=Path, default=root / "data/rolling_stone_80_xai_train_overlap.csv")
    parser.add_argument("--min-coverage", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be greater than 0 and at most 1")
    compare(args.subset_dir, args.train_csv, args.output, args.min_coverage)


if __name__ == "__main__":
    main()
