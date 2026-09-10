"""Deterministic XAI-Lyricist pronunciation scaffolding (prosodic 1.x)."""

from functools import lru_cache


def syllable_features(syllable):
    ipa = str(syllable)
    return {
        "ipa": ipa,
        "stress": "strong" if "'" in ipa else "substrong" if "`" in ipa else "weak",
        "length": "long" if "ː" in ipa else "short",
    }


@lru_cache(maxsize=1)
def backend():
    try:
        import prosodic
    except ImportError as exc:
        raise RuntimeError(
            "IPA preparation requires prosodic 1.6.2; install the ipa extra"
        ) from exc
    if not callable(getattr(prosodic.Text("hello", lang="en"), "words", None)):
        raise RuntimeError("IPA preparation requires the prosodic 1.x API")
    return prosodic


def parse_words(text):
    """Keep IPA counts independent of DALI's sung syllabification, as in the original."""
    text = text.translate(str.maketrans({quote: "'" for quote in "‘’ʼ‛‚′‵ꞌʹʻ"}))
    parsed = backend().Text(
        text.replace("cuz", "cause").replace("-", " ").replace(".", ""), lang="en"
    )
    words = []
    for word in parsed.words():
        syllables = [syllable_features(s) for s in word.syllables()]
        if "?" in word.token or (any(c.isalpha() for c in word.token) and not syllables):
            raise ValueError(f"No IPA pronunciation for {word.token!r}")
        words.append({"text": word.token, "syllables": syllables})
    if not words or not any(w["syllables"] for w in words):
        raise ValueError("No IPA syllables in line")
    return words
