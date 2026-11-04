"""Isolate whole written-score English underlays, prioritizing one vocal line.

This is a conservative automatic screen, not a ground-truth certification or a
training preparer. All pitched attacks in the selected voice are retained; no lyric windows,
inferred melismas, pronunciation repair, or note-count limits are used.
The previous data/pdmx-audit/scores.jsonl supplies a fast candidate prefilter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from zipfile import ZipFile

POLICY = "pdmx_whole_score_english_underlay_v2"
STEPS = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)
FUNCTION_WORDS = set(
    "a an the and or but if then than as at by for from in into of on to with "
    "is are was were be been being am I you he she it we they me him her us them "
    "my your his its our their this that these those not no so do does did have "
    "has had will would shall should can could all who what when where why how".lower().split()
)
VOCAL_NAME = re.compile(
    r"\b(?:voice|voices|vocals?|singer|choir|chorus|soprano|sopranos|baritone|"
    r"contralto|mezzo|cantus|canto)\b|^(?:alto|tenor|bass|treble|[satb][12]?)$",
    re.I,
)
WORDS = re.compile(r"[a-z]+(?:'[a-z]+)*", re.I)
# Pronunciation dictionaries include surnames and bound fragments such as "ly".
# Keep ordinary short English words, interjections and solfege; review the rest.
SHORT_WORDS = {w for w in FUNCTION_WORDS if len(w) <= 2} | set(
    "o oh ah eh er ha ho hi hm mm uh um aw ay ow oy yo ye lo ma pa mo go up "
    "re mi fa la ti si sh st ok ox".split()
)


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return text.translate(str.maketrans({c: "'" for c in "‘’ʼ‛′ʻ"})).strip()


def word_key(text):
    text = normalize(text).lower()
    return re.sub(r"^[^a-z]+|[^a-z']+$", "", text).strip("'")


def ratio(value):
    return [value.numerator, value.denominator]


def lyric_record(elem):
    return {
        "text": normalize("".join(t.text or "" for t in elem.findall("text"))),
        "syllabic": elem.findtext("syllabic", "single"),
        "extend": [e.get("type", "start") for e in elem.findall("extend")],
        "elision": elem.find("elision") is not None,
        "time_only": elem.get("time-only"),
    }


def read_score(path):
    with ZipFile(path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        member = next(
            e.get("full-path") for e in container.iter() if e.tag.rsplit("}", 1)[-1] == "rootfile"
        )
        root = ET.fromstring(archive.read(member))
    return parse_score(root)


def parse_score(root):
    """Read exact score time, checking complete (part, voice) across all staves."""
    for elem in root.iter():
        elem.tag = elem.tag.rsplit("}", 1)[-1]
    if root.tag != "score-partwise":
        raise ValueError("Expected score-partwise")
    parts = {}
    for p in root.findall("part-list/score-part"):
        name = " ".join(p.findtext("part-name", "").split())
        programs = [int(e.text) for e in p.findall("midi-instrument/midi-program")]
        parts[p.get("id")] = {
            "name": name,
            "vocal_hint": bool(VOCAL_NAME.search(name)) or bool({53, 54, 55} & set(programs)),
        }
    voices = defaultdict(list)
    for part in root.findall("part"):
        part_id, divisions, nominal, offset, transpose = part.get("id"), 1, None, Fraction(0), 0
        for measure_index, measure in enumerate(part.findall("measure")):
            cursor = previous_start = maximum = Fraction(0)
            for note_index, elem in enumerate(measure):
                if elem.tag == "attributes":
                    divisions = int(elem.findtext("divisions", str(divisions)))
                    if divisions <= 0:
                        raise ValueError("Nonpositive divisions")
                    time = elem.find("time")
                    if time is not None:
                        beats = time.findall("beats")
                        types = time.findall("beat-type")
                        nominal = (
                            sum(
                                Fraction(sum(map(int, b.text.split("+"))) * 4, int(t.text))
                                for b, t in zip(beats, types, strict=True)
                            )
                            if beats
                            else None
                        )
                    trans = elem.find("transpose")
                    if trans is not None:
                        transpose = int(trans.findtext("chromatic", "0")) + 12 * int(
                            trans.findtext("octave-change", "0")
                        )
                    continue
                if elem.tag in ("backup", "forward"):
                    delta = Fraction(elem.findtext("duration", "0")) / divisions
                    cursor += -delta if elem.tag == "backup" else delta
                    maximum = max(maximum, cursor)
                    if cursor < 0:
                        raise ValueError("Negative within-measure cursor")
                    continue
                if elem.tag != "note":
                    continue
                grace, chord = (elem.find(t) is not None for t in ("grace", "chord"))
                start = previous_start if chord else cursor
                duration = Fraction(elem.findtext("duration", "0")) / divisions
                if not chord:
                    previous_start = start
                if not grace and not chord:
                    cursor += duration
                maximum = max(maximum, cursor, start + duration)
                pitch = elem.find("pitch")
                midi = None
                if pitch is not None:
                    midi = (
                        (int(pitch.findtext("octave")) + 1) * 12
                        + STEPS[pitch.findtext("step")]
                        + Fraction(pitch.findtext("alter", "0"))
                        + transpose
                    )
                lyrics = defaultdict(list)
                duplicates = 0
                for lyric in elem.findall("lyric"):
                    number, record = lyric.get("number", "1"), lyric_record(lyric)
                    if record in lyrics[number]:
                        duplicates += 1
                    else:
                        lyrics[number].append(record)
                ties = {e.get("type") for e in elem.findall("tie")}
                if not ties:
                    ties = {
                        e.get("type")
                        for e in elem.findall("notations/tied")
                        if e.get("type") in ("start", "stop")
                    }
                key = part_id, elem.findtext("voice", "1")
                voices[key].append(
                    {
                        "id": f"{part_id}:m{measure_index}:n{note_index}",
                        "measure_index": measure_index,
                        "measure_number": measure.get("number"),
                        "staff": elem.findtext("staff", "1"),
                        "start": offset + start,
                        "end": offset + start + duration,
                        "pitch": midi,
                        "rest": elem.find("rest") is not None,
                        "grace": grace,
                        "ties": ties,
                        "lyrics": dict(lyrics),
                        "duplicate_lyrics": duplicates,
                    }
                )
            span = maximum if measure.get("implicit") == "yes" else max(maximum, nominal or 0)
            offset += span
    return {
        "parts": parts,
        "voices": dict(voices),
        "has_navigation": any(True for _ in root.iter("repeat"))
        or any(True for _ in root.iter("ending"))
        or any(
            any(k in e.attrib for k in ("dacapo", "dalsegno", "tocoda", "fine"))
            for e in root.iter("sound")
        ),
    }


def attacks_for(notes):
    """Tie-merge without dropping IDs; detect cross-bar overlap and broken ties."""
    attacks, problems, opened = [], Counter(), {}
    problems["unpitched_nonrest_note"] += sum(
        n["pitch"] is None and not n["rest"] and not n["grace"] for n in notes
    )
    pitched = sorted(
        (n for n in notes if n["pitch"] is not None and not n["grace"]),
        key=lambda n: (n["start"], n["id"]),
    )
    previous_end = Fraction(-1)
    for note in pitched:
        if note["end"] <= note["start"]:
            problems["invalid_duration"] += 1
        if note["start"] < previous_end:
            problems["polyphony_or_overlap"] += 1
        previous_end = max(previous_end, note["end"])
        pitch = note["pitch"]
        if pitch.denominator != 1:
            problems["microtonal_pitch"] += 1
        if "stop" in note["ties"]:
            previous = opened.pop(pitch, None)
            if previous is None or previous["end"] != note["start"]:
                problems["broken_tie"] += 1
                previous = None
        else:
            previous = None
        if previous is None:
            previous = {
                "pitch": pitch,
                "start": note["start"],
                "end": note["end"],
                "notes": [note],
            }
            attacks.append(previous)
        else:
            previous["notes"].append(note)
            previous["end"] = note["end"]
        if "start" in note["ties"]:
            if pitch in opened:
                problems["broken_tie"] += 1
            opened[pitch] = previous
    problems["unclosed_tie"] += len(opened)
    return attacks, +problems


def align_verse(notes, attacks, verse):
    """Require every pitched attack to have explicit text, extension, or ties."""
    syllables, problems, active, extending = [], Counter(), None, False
    for note in notes:
        records = note["lyrics"].get(verse, [])
        if records and (note["pitch"] is None or note["grace"]):
            if any(r["text"] for r in records):
                problems["text_on_nonpitched_or_grace_note"] += 1
    for index, attack in enumerate(attacks):
        head = attack["notes"][0]
        records = head["lyrics"].get(verse, [])
        if len(records) > 1:
            problems["conflicting_lyrics_on_note"] += 1
        record = records[0] if records else None
        tail_records = []
        for tail in attack["notes"][1:]:
            tail_records.extend(tail["lyrics"].get(verse, []))
            if any(r["text"] for r in tail["lyrics"].get(verse, [])):
                problems["text_on_tie_continuation"] += 1
        if record and (record["elision"] or record["time_only"]):
            problems["elision_or_conditional_lyric"] += 1
        if record and record["text"]:
            text = re.sub(r"^\d+[.)]\s*", "", record["text"])
            if record["syllabic"] in ("begin", "middle", "end"):
                text = text.strip("- \u00ad")
            active = {
                "text": text,
                "syllabic": record["syllabic"],
                "attack_indices": [],
                "note_ids": [],
            }
            syllables.append(active)
            extending = "start" in record["extend"] or "continue" in record["extend"]
        else:
            explicit = record and any(t in ("start", "continue", "stop") for t in record["extend"])
            if active is None or not (extending or explicit):
                problems["unexplained_untexted_attack"] += 1
                continue
            if index and attacks[index - 1]["end"] != attack["start"]:
                problems["extension_across_gap"] += 1
            if explicit:
                extending = "stop" not in record["extend"]
        active["attack_indices"].append(index)
        active["note_ids"].extend(n["id"] for n in attack["notes"])
        if record and "stop" in record["extend"]:
            extending = False
        for tail_record in tail_records:
            if "stop" in tail_record["extend"]:
                extending = False
            elif tail_record["extend"]:
                extending = True
    words, pending = [], None
    for index, syllable in enumerate(syllables):
        kind, text = syllable["syllabic"], syllable["text"]
        if len(text.split()) != 1 or "-" in text:
            problems["multiword_or_literal_hyphen_syllable"] += 1
        if kind in ("single", "begin"):
            if pending is not None:
                problems["broken_word_chain"] += 1
                words.append(pending)
            pending = {"text": text, "syllable_indices": [index]}
        elif kind in ("middle", "end"):
            if pending is None:
                problems["broken_word_chain"] += 1
                pending = {"text": "", "syllable_indices": []}
            pending["text"] += text
            pending["syllable_indices"].append(index)
        else:
            problems["unsupported_syllabic"] += 1
            continue
        if kind in ("single", "end"):
            words.append(pending)
            pending = None
    if pending is not None:
        words.append(pending)
        problems["broken_word_chain"] += 1
    assigned = sum(len(s["attack_indices"]) for s in syllables)
    return {
        "verse": verse,
        "text": " ".join(w["text"] for w in words),
        "words": words,
        "syllables": syllables,
        "attack_coverage": assigned / max(len(attacks), 1),
        "problems": +problems,
    }


class EnglishCheck:
    def __init__(self):
        import cmudict
        from langdetect import DetectorFactory, detect_langs

        from prosodia_lyricist.ipa import _historical_lexicon

        DetectorFactory.seed = 1234
        self.detect = detect_langs
        self.cmudict = cmudict.dict()
        self.historical, _, _ = _historical_lexicon()
        self.count_cache, self.language_cache = {}, {}

    def counts(self, key):
        if key not in self.count_cache:
            variants = [sum(p[-1:].isdigit() for p in pron) for pron in self.cmudict.get(key, [])]
            historical = self.historical.get(key)
            default = (
                historical.count(".") + 1 if historical else (variants[0] if variants else None)
            )
            counts = set(variants)
            if default:
                counts.add(default)
            self.count_cache[key] = sorted(counts), default
        return self.count_cache[key]

    def preview(self, text):
        words = WORDS.findall(normalize(text).lower())
        return len(words) >= 10 and sum(bool(self.counts(w)[0]) for w in words) / len(words) >= 0.7

    def assess(self, alignment, min_words):
        unknown, mismatches, default_mismatches = [], [], []
        fragments = []
        keys = []
        for word in alignment["words"]:
            key = word_key(word["text"])
            keys.append(key)
            if len(key) == 2 and key not in SHORT_WORDS:
                # Al/Ed are ordinary personal names, but lowercase "al"/"ed"
                # can be fragments of always, sounded, etc.
                if key not in ("al", "ed") or not word["text"].lstrip("\"'")[0].isupper():
                    fragments.append(word["text"])
            counts, default = self.counts(key)
            word["dictionary_syllable_counts"] = counts
            word["default_syllable_count"] = default
            observed = len(word["syllable_indices"])
            if not counts:
                unknown.append(word["text"])
            elif observed not in counts:
                mismatches.append({"word": word["text"], "score": observed, "dictionary": counts})
            if default is not None and observed != default:
                default_mismatches.append(word["text"])
        problems = alignment["problems"]
        if len(keys) < min_words:
            problems["too_few_words"] += 1
        if unknown:
            problems["unknown_words"] += len(unknown)
        if mismatches:
            problems["syllable_count_mismatch"] += len(mismatches)
        if fragments:
            problems["suspected_word_fragments"] += len(fragments)
        function_words = set(keys) & FUNCTION_WORDS
        top, chunks = None, []
        # Only spend detector time on substantially English-looking full texts.
        coverage = 1 - len(unknown) / max(len(keys), 1)
        if len(keys) >= min_words and coverage >= 0.9:
            text = " ".join(keys)
            if text not in self.language_cache:
                try:
                    top = self.detect(text)[0]
                    spans = [keys[i : i + 60] for i in range(0, len(keys), 60)]
                    if len(spans) > 1 and len(spans[-1]) < 20:
                        spans[-2].extend(spans.pop())
                    chunks = (
                        [self.detect(" ".join(span))[0] for span in spans]
                        if top.lang == "en"
                        else []
                    )
                    self.language_cache[text] = (
                        {"language": top.lang, "probability": top.prob},
                        [{"language": c.lang, "probability": c.prob} for c in chunks],
                    )
                except Exception as exc:
                    self.language_cache[text] = ({"error": str(exc)}, [])
            top, chunks = self.language_cache[text]
        english = bool(
            top
            and top.get("language") == "en"
            and top["probability"] >= 0.99
            and len(function_words) >= 4
            and all(c["language"] == "en" and c["probability"] >= 0.90 for c in chunks)
        )
        if not english:
            problems["english_not_confirmed"] += 1
        return {
            "english": english,
            "language": top,
            "language_chunks": chunks,
            "lexical_coverage": coverage,
            "unknown_words": unknown,
            "syllable_mismatches": mismatches,
            "default_count_mismatches": default_mismatches,
            "suspected_word_fragments": fragments,
        }


def credible_vocal_voices(score):
    result = set()
    for key, notes in score["voices"].items():
        pitched = [n for n in notes if n["pitch"] is not None and not n["grace"]]
        text_notes = sum(
            any(any(c.isalpha() for c in r["text"]) for rs in n["lyrics"].values() for r in rs)
            for n in pitched
        )
        if text_notes >= 2 or (len(pitched) >= 20 and score["parts"][key[0]]["vocal_hint"]):
            result.add(key)
    return result


def export_voice(key, attacks, verses, score):
    return {
        "part": key[0],
        "voice": key[1],
        "part_name": score["parts"][key[0]]["name"],
        "attacks": [
            {
                "pitch": int(a["pitch"]),
                "onset_quarters": ratio(a["start"]),
                "duration_quarters": ratio(a["end"] - a["start"]),
                "note_ids": [n["id"] for n in a["notes"]],
                "measures": [n["measure_index"] for n in a["notes"]],
            }
            for a in attacks
        ],
        "verses": verses,
    }


def write_csv(path, rows):
    fields = [
        "score_id",
        "title",
        "artist",
        "work_group",
        "mxl",
        "metadata",
        "category",
        "part",
        "voice",
        "part_name",
        "verse",
        "words",
        "syllables",
        "attacks",
        "default_count_mismatches",
        "vocal_voice_count",
        "has_navigation",
        "upstream_deduplicated",
        "dali_splits",
        "text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)


def isolate(args):
    started, checker = monotonic(), EnglishCheck()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    with args.audit.open() as handle:
        for line in handle:
            row = json.loads(line)
            selected = [
                s
                for s in row["streams"]
                if s["alphabetic_lyric_count"] >= 20
                and s["word_count"] >= 10
                and s["pitched_notes_in_voice"] >= 20
                and not s["within_measure_overlap"]
                and checker.preview(s["text_preview"])
            ]
            if selected:
                row["candidate_voices"] = sorted({(s["part"], s["voice"]) for s in selected})
                candidates.append(row)
    candidates.sort(
        key=lambda r: (
            r["flags"]["multiple_lyric_voices"],
            hashlib.sha256(r["mxl"].encode()).digest(),
        )
    )
    parent_manifest = None
    if args.accepted_from:
        source = args.accepted_from.resolve()
        if source == output:
            raise ValueError("accepted-from must differ from output")
        accepted_paths = {json.loads(line)["mxl"] for line in (source / "songs.jsonl").open()}
        candidates = [r for r in candidates if r["mxl"] in accepted_paths]
        parent_manifest = json.loads((source / "manifest.json").read_text())
        (output / "input_screen_manifest.json").write_text(
            json.dumps(parent_manifest, indent=2) + "\n"
        )
    if args.limit:
        candidates = candidates[: args.limit]
    selected_paths = {r["path"] for r in candidates}
    with (args.pdmx_root / "PDMX.csv").open(newline="") as handle:
        catalog = {r["path"]: r for r in csv.DictReader(handle) if r["path"] in selected_paths}
    dali_path = args.audit.parent / "dali_identity_overlap.json"
    dali = {}
    if dali_path.exists():
        for match in json.loads(dali_path.read_text())["matches"]:
            dali[match["mxl"]] = ",".join(sorted({d["split"] for d in match["dali"]}))
    totals, reasons, records, csv_rows = Counter(), Counter(), [], []
    review_rows = []
    with (
        (output / "screening.jsonl").open("w") as screening,
        (output / "songs.jsonl").open("w") as songs,
    ):
        for index, candidate in enumerate(candidates, 1):
            result = {"mxl": candidate["mxl"], "title": candidate["title"], "voices": []}
            accepted = []
            try:
                score = read_score(args.pdmx_root / candidate["mxl"])
                vocal = credible_vocal_voices(score)
                category = "single_vocal" if len(vocal) == 1 else "multi_vocal"
                for key in map(tuple, candidate["candidate_voices"]):
                    notes = score["voices"][key]
                    attacks, voice_problems = attacks_for(notes)
                    verses = sorted({v for n in notes for v in n["lyrics"]})
                    accepted_verses = []
                    for verse in verses:
                        alignment = align_verse(notes, attacks, verse)
                        alignment["problems"].update(voice_problems)
                        assessment = checker.assess(alignment, args.min_words)
                        problems = dict(+alignment["problems"])
                        diagnostics = {
                            "part": key[0],
                            "voice": key[1],
                            "verse": verse,
                            "words": len(alignment["words"]),
                            "syllables": len(alignment["syllables"]),
                            "attacks": len(attacks),
                            "coverage": alignment["attack_coverage"],
                            "problems": problems,
                            **assessment,
                        }
                        result["voices"].append(diagnostics)
                        totals["verse_candidates"] += 1
                        if problems:
                            reasons.update(problems.keys())
                            reviewable = {
                                "unknown_words",
                                "syllable_count_mismatch",
                                "unexplained_untexted_attack",
                                "extension_across_gap",
                                "text_on_tie_continuation",
                                "suspected_word_fragments",
                            }
                            if (
                                assessment["english"]
                                and assessment["lexical_coverage"] >= 0.95
                                and alignment["attack_coverage"] >= 0.95
                                and set(problems) <= reviewable
                            ):
                                review_rows.append(
                                    {
                                        "title": candidate["title"],
                                        "mxl": candidate["mxl"],
                                        "part": key[0],
                                        "voice": key[1],
                                        "verse": verse,
                                        "category": category,
                                        "coverage": alignment["attack_coverage"],
                                        "problems": json.dumps(problems, sort_keys=True),
                                        "unknown_words": json.dumps(assessment["unknown_words"]),
                                        "suspected_word_fragments": json.dumps(
                                            assessment["suspected_word_fragments"]
                                        ),
                                        "syllable_mismatches": json.dumps(
                                            assessment["syllable_mismatches"]
                                        ),
                                        "text": alignment["text"],
                                    }
                                )
                            continue
                        alignment["problems"] = {}
                        alignment["checks"] = assessment
                        accepted_verses.append(alignment)
                    if accepted_verses:
                        accepted.append(export_voice(key, attacks, accepted_verses, score))
                if accepted:
                    row = catalog[candidate["path"]]
                    record = {
                        "id": candidate["path"].split("/")[-1].removesuffix(".json"),
                        "score_id": candidate["score_id"],
                        "title": candidate["title"],
                        "artist": row["artist_name"],
                        "mxl": candidate["mxl"],
                        "metadata": candidate["metadata"],
                        "best_path": candidate["best_path"],
                        "song_catalog_id": candidate["song_catalog_id"],
                        "category": category,
                        "vocal_voice_count": len(vocal),
                        "validated_vocal_voice_count": len(accepted),
                        "has_navigation": score["has_navigation"],
                        "upstream_deduplicated": candidate["deduplicated"],
                        "dali_splits": dali.get(candidate["mxl"], ""),
                        "voices": accepted,
                    }
                    records.append(record)
                    songs.write(json.dumps(record, ensure_ascii=False) + "\n")
                    totals[category + "_scores"] += 1
                    totals[category + "_verses"] += sum(len(v["verses"]) for v in accepted)
                    for voice in accepted:
                        for verse in voice["verses"]:
                            csv_rows.append(
                                {
                                    **{
                                        k: record[k]
                                        for k in (
                                            "score_id",
                                            "title",
                                            "artist",
                                            "mxl",
                                            "metadata",
                                            "category",
                                            "vocal_voice_count",
                                            "has_navigation",
                                            "upstream_deduplicated",
                                            "dali_splits",
                                        )
                                    },
                                    "part": voice["part"],
                                    "voice": voice["voice"],
                                    "part_name": voice["part_name"],
                                    "verse": verse["verse"],
                                    "words": len(verse["words"]),
                                    "syllables": len(verse["syllables"]),
                                    "attacks": len(voice["attacks"]),
                                    "text": verse["text"],
                                    "default_count_mismatches": len(
                                        verse["checks"]["default_count_mismatches"]
                                    ),
                                }
                            )
                result["category"] = category
                result["accepted"] = bool(accepted)
            except (OSError, ValueError, KeyError, ET.ParseError) as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                totals["score_errors"] += 1
            screening.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 250 == 0:
                print(
                    f"{index}/{len(candidates)} scores; accepted {len(records)}; "
                    f"{monotonic() - started:.1f}s",
                    flush=True,
                )
    # Union upstream work groups and explicit MuseScore song identities before ranking.
    parent = {r["mxl"]: r["mxl"] for r in records}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for record in records:
        identities = [("best_path", record["best_path"])]
        if record["song_catalog_id"] is not None:
            identities.append(("song_id", record["song_catalog_id"]))
        for identity in identities:
            if identity in seen:
                a, b = sorted((root(record["mxl"]), root(seen[identity])))
                parent[b] = a
            seen[identity] = record["mxl"]
    for row in csv_rows:
        row["work_group"] = root(row["mxl"])

    def rank(row):
        return (
            row["category"] != "single_vocal",
            row["default_count_mismatches"],
            row["has_navigation"],
            not row["upstream_deduplicated"],
            -row["words"],
            row["mxl"],
            row["verse"],
        )

    representatives = {}
    for row in sorted(csv_rows, key=rank):
        representatives.setdefault(row["work_group"], row)
    for name in ("single_vocal", "multi_vocal"):
        rows = [r for r in csv_rows if r["category"] == name]
        chosen = [r for r in representatives.values() if r["category"] == name]
        write_csv(output / f"{name}_all_verses.csv", rows)
        write_csv(output / f"{name}_songs.csv", chosen)
        totals[name + "_work_representatives"] = len(chosen)
    if review_rows:
        with (output / "review_verses.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
            writer.writeheader()
            writer.writerows(review_rows)
    totals["near_miss_review_scores"] = len({row["mxl"] for row in review_rows})
    totals["near_miss_review_verses"] = len(review_rows)
    manifest = {
        "policy": POLICY,
        "scope": "whole written score, selected complete part/voice/verse; repeats not unfolded",
        "partial": args.limit is not None,
        "pdmx_root": str(args.pdmx_root.resolve()),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "min_words": args.min_words,
        "prefiltered_scores": len(candidates),
        "accepted_from": str(args.accepted_from.resolve()) if args.accepted_from else None,
        "parent_screened_scores": parent_manifest["prefiltered_scores"]
        if parent_manifest
        else None,
        "counts": totals,
        "verse_rejection_reasons": reasons,
        "acceptance": [
            "Complete part/voice across all staves is monophonic, with valid contiguous ties.",
            "All pitched attacks map to explicit lyric syllables, ties or explicit melismas; "
            "no guessed notes or crops.",
            "All words reconstruct through valid syllabic chains and have a "
            "dictionary-compatible syllable count.",
            "Every word is in CMUdict or the existing historical English dictionary.",
            "Standalone short fragments such as ly, ve, ou and foreign function words "
            "are held for review even when pronunciation dictionaries contain them.",
            "Whole-text English probability >=0.99 and every 60-word chunk >=0.90, "
            "plus >=4 English function words.",
            "Single-vocal tier additionally has no other line with >=2 lyric notes "
            "or >=20 notes and a vocal name/program hint.",
        ],
        "limitations": [
            "Automatic screening cannot certify transcription correctness; "
            "inspect source scores before training.",
            "Dictionary count compatibility includes pronunciation variants; "
            "default-count mismatches are reported separately.",
            "No full IPA backend run, performance repeat unfolding, "
            "or new rights verification is performed.",
            "Preview/monophony prefilter is conservative and can miss good scores; "
            "multi-vocal tier applies the same underlay checks.",
            "Work groups use identities among accepted scores only and "
            "are not final train/validation/test assignments.",
        ],
        "packages": {p: version(p) for p in ("cmudict", "langdetect")},
        "seconds": round(monotonic() - started, 1),
        "sha256": {},
    }
    for path in output.iterdir():
        if path.is_file() and path.suffix in (".csv", ".jsonl"):
            manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdmx-root", type=Path, default=Path("data/pdmx"))
    parser.add_argument("--audit", type=Path, default=Path("data/pdmx-audit/scores.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/pdmx-whole-song"))
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--accepted-from",
        type=Path,
        help="Tighten an existing screen's accepted scores; preserve its parent manifest",
    )
    args = parser.parse_args()
    if args.min_words < 1 or (args.limit is not None and args.limit < 1):
        parser.error("min-words and limit must be positive")
    isolate(args)


if __name__ == "__main__":
    main()
