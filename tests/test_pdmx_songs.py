"""Conservative whole-score underlay screening must not invent lyric links."""

import xml.etree.ElementTree as ET
from collections import Counter
from types import SimpleNamespace

from scripts.isolate_pdmx_songs import (
    EnglishCheck,
    align_verse,
    attacks_for,
    credible_vocal_voices,
    parse_score,
)


def lyric(text="", kind="single", verse="1", extension=""):
    return (
        f'<lyric number="{verse}"><syllabic>{kind}</syllabic><text>{text}</text>{extension}</lyric>'
    )


def note(text="", *, pitch="C", tie="", voice="1", staff="1", duration=1):
    return (
        f"<note><pitch><step>{pitch}</step><octave>4</octave></pitch>"
        f"<duration>{duration}</duration><voice>{voice}</voice><staff>{staff}</staff>"
        f"{tie}{text}</note>"
    )


def score(measures, name="Voice"):
    return parse_score(
        ET.fromstring(
            '<score-partwise><part-list><score-part id="P1">'
            f"<part-name>{name}</part-name></score-part></part-list><part id='P1'>"
            f"{measures}</part></score-partwise>"
        )
    )


def aligned(parsed, verse="1"):
    notes = parsed["voices"]["P1", "1"]
    attacks, problems = attacks_for(notes)
    return attacks, problems, align_verse(notes, attacks, verse)


def test_cross_bar_tie_preserves_both_note_ids_and_one_syllable():
    parsed = score(
        '<measure number="1"><attributes><divisions>1</divisions>'
        "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
        "<note><rest/><duration>3</duration><voice>1</voice></note>"
        + note(lyric("world"), tie='<tie type="start"/>')
        + "</measure>"
        '<measure number="2">'
        + note(tie='<tie type="stop"/>')
        + note(lyric("shine"), pitch="D")
        + "</measure>"
    )
    attacks, problems, result = aligned(parsed)
    assert not problems and not result["problems"]
    assert len(attacks) == 2
    assert attacks[0]["start"] == 3 and attacks[0]["end"] == 5
    assert len(result["syllables"][0]["note_ids"]) == 2
    assert result["attack_coverage"] == 1


def test_unlabeled_notes_are_not_assumed_to_be_melisma():
    parsed = score("<measure>" + note(lyric("shine")) + note(pitch="D") + "</measure>")
    _, _, result = aligned(parsed)
    assert result["problems"]["unexplained_untexted_attack"] == 1
    assert result["attack_coverage"] == 0.5


def test_explicit_extension_maps_several_attacks_to_one_syllable():
    parsed = score(
        "<measure>"
        + note(lyric("shine", extension="<extend/>"))
        + note(pitch="D")
        + note(lyric(extension='<extend type="stop"/>'), pitch="E")
        + note(lyric("on"), pitch="F")
        + "</measure>"
    )
    _, _, result = aligned(parsed)
    assert not result["problems"]
    assert result["syllables"][0]["attack_indices"] == [0, 1, 2]
    assert result["syllables"][1]["attack_indices"] == [3]


def test_extension_stop_on_tie_tail_does_not_leak_to_next_note():
    parsed = score(
        "<measure>"
        + note(lyric("shine", extension="<extend/>"), tie='<tie type="start"/>')
        + note(lyric(extension='<extend type="stop"/>'), tie='<tie type="stop"/>')
        + note(pitch="D")
        + "</measure>"
    )
    _, _, result = aligned(parsed)
    assert result["problems"]["unexplained_untexted_attack"] == 1


def test_staff_switch_does_not_silently_crop_same_voice_polyphony():
    parsed = score(
        "<measure>"
        + note(lyric("shine"), duration=2)
        + "<backup><duration>2</duration></backup>"
        + note(lyric("on"), staff="2", pitch="D", duration=2)
        + "</measure>"
    )
    assert len(parsed["voices"]) == 1
    _, problems, _ = aligned(parsed)
    assert problems["polyphony_or_overlap"] == 1


def test_pickup_and_changing_divisions_keep_contiguous_tie():
    parsed = score(
        '<measure implicit="yes"><attributes><divisions>2</divisions>'
        "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
        + note(lyric("shine"), duration=2, tie='<tie type="start"/>')
        + "</measure>"
        "<measure><attributes><divisions>3</divisions></attributes>"
        + note(duration=3, tie='<tie type="stop"/>')
        + "</measure>"
    )
    attacks, problems, result = aligned(parsed)
    assert not problems and not result["problems"]
    assert attacks[0]["end"] == 2


def test_duplicate_lyrics_collapse_but_conflicting_same_verse_is_rejected():
    repeated = lyric("shine") + lyric("shine")
    parsed = score("<measure>" + note(repeated) + "</measure>")
    _, _, result = aligned(parsed)
    assert not result["problems"] and result["text"] == "shine"
    parsed = score("<measure>" + note(repeated + lyric("bright")) + "</measure>")
    _, _, result = aligned(parsed)
    assert result["problems"]["conflicting_lyrics_on_note"] == 1


def test_parallel_verses_reconstruct_separately():
    parsed = score(
        "<measure>"
        + note(lyric("hel", "begin") + lyric("sun", "begin", "2"))
        + note(lyric("lo", "end") + lyric("shine", "end", "2"), pitch="D")
        + "</measure>"
    )
    _, _, first = aligned(parsed)
    _, _, second = aligned(parsed, "2")
    assert first["text"] == "hello" and second["text"] == "sunshine"
    assert first["words"][0]["syllable_indices"] == [0, 1]
    assert not first["problems"] and not second["problems"]


def test_another_underlaid_voice_prevents_single_vocal_tier():
    parsed = score(
        "<measure>"
        + note(lyric("shine"))
        + note(lyric("on"))
        + "<backup><duration>2</duration></backup>"
        + note(lyric("sing"), voice="2")
        + note(lyric("out"), voice="2")
        + "</measure>",
        name="Piano",
    )
    assert credible_vocal_voices(parsed) == {("P1", "1"), ("P1", "2")}


def small_checker():
    checker = EnglishCheck.__new__(EnglishCheck)
    checker.cmudict = {
        "fire": [["F", "AY1", "ER0"], ["F", "AY1", "R"]],
        "heaven": [["HH", "EH1", "V", "AH0", "N"]],
        **{w: [["AH1"]] for w in ("the", "and", "is", "in", "my")},
    }
    checker.historical = {"fire": "'faɪ.ə", "heaven": "'he.vən"}
    checker.count_cache, checker.language_cache = {}, {}
    checker.detect = lambda text: [SimpleNamespace(lang="en", prob=1.0)]
    return checker


def test_valid_pronunciation_variant_is_distinct_from_default_count():
    checker = small_checker()
    assert checker.counts("fire") == ([1, 2], 2)
    parsed = score("<measure>" + note(lyric("fire")) + "</measure>")
    _, _, alignment = aligned(parsed)
    checks = checker.assess(alignment, 1)
    assert not checks["syllable_mismatches"]
    assert checks["default_count_mismatches"] == ["fire"]


def test_impossible_word_syllable_count_is_rejected():
    checker = small_checker()
    parsed = score("<measure>" + note(lyric("heaven")) + "</measure>")
    _, _, alignment = aligned(parsed)
    checks = checker.assess(alignment, 1)
    assert checks["syllable_mismatches"] == [{"word": "heaven", "score": 1, "dictionary": [2]}]
    assert alignment["problems"]["syllable_count_mismatch"] == 1


def test_whole_text_english_does_not_override_nonenglish_chunks():
    checker = small_checker()
    checker.detect = lambda text: [
        SimpleNamespace(lang="en" if len(text.split()) > 60 else "fr", prob=1.0)
    ]
    keys = ["the", "and", "is", "in", "my"] * 24
    alignment = {
        "words": [{"text": key, "syllable_indices": [i]} for i, key in enumerate(keys)],
        "problems": Counter(),
    }
    checks = checker.assess(alignment, 20)
    assert checks["language"]["language"] == "en"
    assert not checks["english"]
    assert alignment["problems"]["english_not_confirmed"] == 1


def test_pronunciation_dictionary_fragment_is_not_an_english_word_boundary():
    checker = small_checker()
    checker.historical.update({"sure": "'ʃʊr", "ly": "'laɪ"})
    parsed = score("<measure>" + note(lyric("sure")) + note(lyric("ly"), pitch="D") + "</measure>")
    _, _, alignment = aligned(parsed)
    checks = checker.assess(alignment, 1)
    assert not checks["syllable_mismatches"]
    assert checks["suspected_word_fragments"] == ["ly"]
    assert alignment["problems"]["suspected_word_fragments"] == 1
