"""Keep English catalogue entries for solo voice in 4/4 throughout.

Common time is accepted; cut time, displayed meter overrides, and meter changes
are checked. Small optional notes, alternate cutaway staves, and duplicate
unison notes used to engrave verse rhythms do not create extra vocal lines.
The original catalogue and score files are read only. An audit CSV records every
decision, including scores with no vocal part or no explicit time signature.
"""

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile


DURATIONS = {
    "long": Fraction(4), "breve": Fraction(2), "whole": Fraction(1),
    "half": Fraction(1, 2), "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8), "16th": Fraction(1, 16),
    "32nd": Fraction(1, 32), "64th": Fraction(1, 64),
    "128th": Fraction(1, 128), "256th": Fraction(1, 256),
}


def primary_pitches(chord):
    if chord.findtext("small") == "1" or chord.findtext("visible") == "0":
        return set()
    return {
        int(note.findtext("pitch"))
        for note in chord.findall("Note")
        if note.findtext("small") != "1"
        and note.findtext("visible") != "0"
        and note.findtext("play") != "0"
    }


def meter(time_sig):
    # A custom displayed signature takes precedence over stored playback meter.
    numerator = time_sig.findtext("textN") or time_sig.findtext("sigN")
    denominator = time_sig.findtext("textD") or time_sig.findtext("sigD")
    return f"{numerator}/{denominator}"


def has_polyphony(staff):
    """Find simultaneous independent pitches, allowing duplicate unison notes."""
    for measure in staff.findall("Measure"):
        tuplets = {
            tuplet.get("id"): tuplet
            for tuplet in measure.findall(".//Tuplet")
            if tuplet.get("id") is not None
        }

        def tuplet_scale(identifier):
            if identifier is None:
                return Fraction(1)
            tuplet = tuplets[identifier]
            return Fraction(
                int(tuplet.findtext("normalNotes")),
                int(tuplet.findtext("actualNotes")),
            ) * tuplet_scale(tuplet.findtext("Tuplet"))

        events = []
        for voice_index, voice in enumerate(measure.findall("voice")):
            position = Fraction(0)
            for element in voice:
                if element.tag == "location":
                    if element.find("measures") is not None:
                        raise ValueError("Unexpected cross-measure voice location")
                    position += Fraction(element.findtext("fractions", "0"))
                elif element.tag in ("Chord", "Rest"):
                    if any(
                        child.tag.startswith("grace")
                        or child.tag in ("acciaccatura", "appoggiatura")
                        for child in element
                    ):
                        continue
                    duration_type = element.findtext("durationType")
                    if duration_type == "measure":
                        duration = Fraction(
                            element.findtext("duration") or measure.get("len", "1")
                        )
                    else:
                        duration = DURATIONS[duration_type]
                        dots = int(element.findtext("dots", "0"))
                        duration *= 2 - Fraction(1, 2**dots)
                        duration *= tuplet_scale(element.findtext("Tuplet"))
                    pitches = primary_pitches(element) if element.tag == "Chord" else set()
                    if len(pitches) > 1:
                        return True
                    if pitches:
                        events.append((position, position + duration, pitches, voice_index))
                    position += duration
        for i, (start, end, pitches, voice) in enumerate(events):
            for other_start, other_end, other_pitches, other_voice in events[i + 1 :]:
                if (
                    voice != other_voice
                    and max(start, other_start) < min(end, other_end)
                    and pitches != other_pitches
                ):
                    return True
    return False


def inspect_score(path):
    with ZipFile(path) as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith(".mscx") and "/" not in name
        ]
        if len(names) != 1:
            raise ValueError(f"Expected one score XML in {path}")
        score = ET.fromstring(archive.read(names[0])).find("Score")
    if score is None:
        raise ValueError(f"Missing Score element in {path}")

    staves = score.findall("Staff")
    definitions = [
        (part, staff)
        for part in score.findall("Part")
        for staff in part.findall("Staff")
    ]
    if len(staves) != len(definitions):
        raise ValueError(f"Staff definitions do not match notation in {path}")
    vocals = []
    for staff, (part, definition) in zip(staves, definitions):
        instrument_id = part.findtext("Instrument/instrumentId", "")
        if not instrument_id.startswith("voice.") and staff.find(".//Lyrics") is None:
            continue
        label = part.findtext("trackName", "")
        notes = [chord for chord in staff.findall(".//Chord") if primary_pitches(chord)]
        is_alternative = bool(re.search(r"cutaway|alternative|ossia", label, re.I))
        # The cutaway property alone can also be used on genuine choral staves.
        if definition.findtext("cutaway") == "1":
            directions = [
                element.findtext("text", "").strip().lower()
                for tag in ("Expression", "StaffText")
                for element in staff.findall(f".//{tag}")
            ]
            is_alternative |= any(text in ("or", "oppure") for text in directions)
        vocals.append((staff, label, is_alternative or not notes))

    primary = [(staff, label) for staff, label, alternate in vocals if not alternate]
    ignored = [staff.get("id") for staff, _, alternate in vocals if alternate]
    signatures = score.findall("./Staff//TimeSig")
    meters = sorted({meter(signature) for signature in signatures})
    reasons = []
    if not primary:
        reasons.append("no_independent_vocal_line")
    elif len(primary) > 1:
        reasons.append("multiple_vocal_lines")
    elif has_polyphony(primary[0][0]):
        reasons.append("vocal_polyphony")
    if not signatures:
        reasons.append("no_explicit_time_signature")
    elif meters != ["4/4"]:
        reasons.append("not_4_4_throughout")
    return {
        "keep": not reasons,
        "reason": "; ".join(reasons) or "solo_voice_4_4_throughout",
        "independent_vocal_staves": len(primary),
        "vocal_staff_ids": "; ".join(staff.get("id", "") for staff, _ in primary),
        "ignored_alternative_staff_ids": "; ".join(ignored),
        "time_signatures": "; ".join(meters),
        "common_time_used": any(s.findtext("subtype") == "1" for s in signatures),
    }


def filter_catalogue(catalogue, output, audit):
    if catalogue.resolve() in (output.resolve(), audit.resolve()):
        raise ValueError("Output paths must differ from the original catalogue")
    if output.resolve() == audit.resolve():
        raise ValueError("Filtered and audit CSV paths must differ")
    with catalogue.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    a, b = fields.index("url"), fields.index("filename")
    fields[a], fields[b] = fields[b], fields[a]
    decisions = [
        inspect_score(catalogue.parent / (row.get("local_file") or row["filename"]))
        for row in rows
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row, result in zip(rows, decisions) if result["keep"])
    with audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + list(decisions[0]))
        writer.writeheader()
        writer.writerows(dict(row, **result) for row, result in zip(rows, decisions))
    print(json.dumps({
        "input_songs": len(rows),
        "retained": sum(result["keep"] for result in decisions),
        "removed": sum(not result["keep"] for result in decisions),
        "decisions": dict(Counter(result["reason"] for result in decisions)),
        "output": str(output), "audit": str(audit),
    }, indent=2))


def main():
    root = Path(__file__).resolve().parents[1] / "data/open-lieder"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=root / "english_catalogue.csv")
    parser.add_argument("--output", type=Path, default=root / "english_catalogue_solo_4_4.csv")
    parser.add_argument("--audit", type=Path, default=root / "english_catalogue_filter_audit.csv")
    args = parser.parse_args()
    filter_catalogue(args.catalogue, args.output, args.audit)


if __name__ == "__main__":
    main()
