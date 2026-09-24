"""Read DALI's horizontal annotations and align sung syllables to lyric lines.

The official .gz files pickle a DALI.Annotations object. Only that data holder
and the NumPy scalar types used by the release are needed here; importing the
DALI package would also bring in its unrelated audio downloading stack.
"""

import codecs
import gzip
import json
import math
import pickle
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from numpy._core.multiarray import scalar as numpy_scalar
except ImportError:  # NumPy 1.x stores the scalar constructor under numpy.core.
    from numpy.core.multiarray import scalar as numpy_scalar


class AnnotationError(ValueError):
    """An annotation cannot be aligned without guessing."""


class _Annotation:
    pass


class _DALIUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("DALI.Annotations", "Annotations"): _Annotation,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): numpy_scalar,
            ("numpy._core.multiarray", "scalar"): numpy_scalar,
            ("_codecs", "encode"): codecs.encode,
        }
        try:
            return allowed[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"Unsupported DALI pickle type: {module}.{name}")


def read_annotation(path):
    """Return info and annotations from an official .gz or DALI JSON export."""
    path = Path(path)
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        info, annotations = data["info"], data["annotations"]
    else:
        with gzip.open(path, "rb") as handle:
            entry = _DALIUnpickler(handle, encoding="latin1").load()
        info, annotations = entry.info, entry.annotations
    if annotations.get("type") != "horizontal":
        raise AnnotationError("Expected horizontal DALI annotations")
    for level in ("notes", "words", "lines"):
        if not isinstance(annotations["annot"].get(level), list):
            raise AnnotationError(f"Missing annotation level: {level}")
    return info, annotations["annot"]


def normalize_text(text):
    return " ".join(str(text).replace("’", "'").replace("‘", "'").split())


def interval(item):
    try:
        start, end = map(float, item["time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationError("Invalid time interval") from exc
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise AnnotationError("Non-finite, negative, or non-positive time interval")
    return start, end


def parent_index(item, size):
    index = item.get("index")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise AnnotationError("Non-integer parent index")
    if not 0 <= index < size:
        raise AnnotationError("Parent index out of range")
    return int(index)


def sung_syllables(notes):
    """Merge '~' melisma continuations; rests do not add sung duration.

    The DALI note text describes sung syllables, which need not match dictionary
    pronunciations. Empty/tilde-prefixed continuation notes extend the preceding
    syllable within the same word. An orphan continuation is rejected.
    """
    syllables = []
    last_end = -math.inf
    for note in notes:
        start, end = interval(note)
        if start < last_end - 1e-6:
            raise AnnotationError("Overlapping or unordered notes within a word")
        last_end = end
        text = normalize_text(note.get("text", ""))
        if not text or text.startswith("~"):
            if not syllables:
                raise AnnotationError("Melisma continuation without a preceding syllable")
            syllables[-1]["text"] += text.lstrip("~")
            syllables[-1]["end"] = end
            syllables[-1]["duration"] += end - start
            syllables[-1]["note_count"] += 1
        else:
            syllables.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "note_count": 1,
                }
            )
    if not syllables:
        raise AnnotationError("Word has no notes")
    return syllables


@lru_cache(maxsize=1)
def pronunciation_dictionary():
    import cmudict

    return cmudict.dict()


def lexical_stresses(word, count, pronunciations=None):
    """Use a matching CMU pronunciation, otherwise explicitly mark unknown."""
    if pronunciations is None:
        pronunciations = pronunciation_dictionary()
    key = re.sub(r"^[^\w']+|[^\w']+$", "", normalize_text(word).lower())
    for phones in pronunciations.get(key, []):
        stresses = [phone[-1] for phone in phones if phone[-1:] in ("0", "1", "2")]
        if len(stresses) == count:
            return [{"0": "weak", "1": "strong", "2": "substrong"}[s] for s in stresses]
    return ["unknown"] * count


def aligned_words(words, notes_by_word, line_start, line_end):
    """Validate original bounds, then attach empty tilde-only parents locally.

    Some DALI exports give melisma notes their own empty word. Only explicit
    tilde-only continuations may extend the previous word in the same line;
    never infer missing lyrics or concatenate neighboring text fragments.
    """
    aligned = []
    last_end = line_start
    for word_id, word in words:
        start, end = interval(word)
        if start < last_end - 1e-6 or end > line_end + 1e-6:
            raise AnnotationError("Word lies outside its line or overlaps the previous word")
        last_end = end
        notes = notes_by_word[word_id]
        for note in notes:
            note_start, note_end = interval(note)
            if note_start < start - 1e-6 or note_end > end + 1e-6:
                raise AnnotationError("Notes lie outside their parent word")
        raw_text = word.get("text")
        word_text = normalize_text(raw_text) if isinstance(raw_text, str) else ""
        if not word_text:
            if not notes or not all(
                (text := normalize_text(n.get("text", ""))) and not text.strip("~")
                for n in notes
            ):
                raise AnnotationError("Empty word text")
            if not aligned:
                raise AnnotationError("Melisma continuation without a preceding syllable")
            aligned[-1][1].extend(notes)
        else:
            aligned.append((word_text, list(notes)))
    return aligned


def extract_lines(info, annot, *, max_syllables=64, stress_source="lexical"):
    """Return usable line records and explicit rejection reasons.

    Parent indices, not timestamp proximity or paragraph ids, define membership.
    One malformed line does not hide otherwise usable lines in the same song.
    """
    if stress_source not in ("ipa", "lexical", "unknown"):
        raise ValueError("stress_source must be ipa, lexical or unknown")
    words_by_line, notes_by_word = defaultdict(list), defaultdict(list)
    for word_id, word in enumerate(annot["words"]):
        words_by_line[parent_index(word, len(annot["lines"]))].append((word_id, word))
    for note in annot["notes"]:
        notes_by_word[parent_index(note, len(annot["words"]))].append(note)
    records, rejected = [], []
    for line_id, line in enumerate(annot["lines"]):
        try:
            line_start, line_end = interval(line)
            syllables, text, scaffold_words = [], [], []
            for word_text, notes in aligned_words(
                words_by_line[line_id], notes_by_word, line_start, line_end
            ):
                sung = sung_syllables(notes)
                stresses = (
                    lexical_stresses(word_text, len(sung))
                    if stress_source == "lexical"
                    else ["unknown"] * len(sung)
                )
                for syllable, stress in zip(sung, stresses):
                    syllable["stress"] = stress
                syllables.extend(sung)
                text.append(word_text)
                scaffold_words.append({"text": word_text, "syllable_count": len(sung)})
            # Relative duration needs no tempo, beat grid, or bar-line estimate.
            if not syllables:
                raise AnnotationError("Empty line")
            mean_duration = sum(s["duration"] for s in syllables) / len(syllables)
            for syllable in syllables:
                syllable["length"] = "long" if syllable["duration"] > mean_duration else "short"
            sung_template = syllables
            if stress_source == "ipa":
                from .ipa import parse_words

                try:
                    parsed = parse_words(" ".join(text))
                except ValueError as exc:
                    raise AnnotationError(str(exc)) from exc
                syllables = [s for word in parsed for s in word["syllables"]]
                scaffold_words = [
                    {"text": w["text"], "syllable_count": len(w["syllables"])} for w in parsed
                ]
                text = [w["text"] for w in parsed]
            if not 1 <= len(syllables) <= max_syllables:
                raise AnnotationError("Empty line or syllable limit exceeded")
            records.append(
                {
                    "id": f"{info['id']}:{line_id}",
                    "song_id": info["id"],
                    "line_id": line_id,
                    "title": normalize_text(info.get("title", "")),
                    "text": " ".join(text),
                    "start": line_start,
                    "end": line_end,
                    "syllables": syllables,
                    "words": scaffold_words,
                    "sung_syllables": sung_template,
                }
            )
        except AnnotationError as exc:
            rejected.append({"song_id": info["id"], "line_id": line_id, "reason": str(exc)})
    return records, rejected
