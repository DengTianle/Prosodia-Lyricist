"""Deterministic XAI-Lyricist IPA features using the modern prosodic API."""

from functools import lru_cache


def syllable_features(syllable):
    # Modern Syllable.__str__ displays orthography, not IPA.
    ipa = syllable if isinstance(syllable, str) else syllable.ipa
    return {
        "ipa": ipa,
        "stress": (
            "strong"
            if any(mark in ipa for mark in ("'", "ˈ"))
            else "substrong"
            if any(mark in ipa for mark in ("`", "ˌ"))
            else "weak"
        ),
        "length": "long" if "ː" in ipa else "short",
    }


@lru_cache(maxsize=1)
def backend():
    try:
        import prosodic
    except ImportError as exc:
        raise RuntimeError(
            "IPA preparation requires modern prosodic; install the ipa extra"
        ) from exc
    try:
        # Materialize lazy words too, so missing resources fail before output creation.
        list(prosodic.Text("hello", lang="en", syntax=False).wordtokens)
    except (AttributeError, LookupError) as exc:
        raise RuntimeError(
            "IPA preparation requires prosodic>=3.10 and its pronunciation/tokenizer "
            "resources. Install the ipa extra and NLTK punkt/punkt_tab data."
        ) from exc
    return prosodic


def parse_words(text):
    """Select the first pronunciation per word, independently of DALI sung counts."""
    try:
        return _parse_words(text)
    except (OSError, LookupError) as exc:
        # Environment failures must abort preparation, not mark songs as bad data.
        raise RuntimeError(f"IPA backend resource/cache failure: {exc}") from exc


def _parse_words(text):
    text = text.translate(str.maketrans({quote: "'" for quote in "‘’ʼ‛‚′‵ꞌʹʻ"}))
    parsed = backend().Text(
        text.replace("cuz", "cause").replace("-", " ").replace(".", ""),
        lang="en",
        syntax=False,
    )
    words = []
    for word in parsed.wordtokens:
        # Flattening all wordforms would count alternative pronunciations twice.
        form = word.wordtype.form
        syllables = [syllable_features(s) for s in form.syllables] if form is not None else []
        token = word.txt.strip()
        if not token:
            continue
        # Punctuation tokens (including '?') have no pronunciation by design.
        # Retain them with zero syllables, just like commas and exclamation marks.
        if any(c.isalpha() for c in token) and not syllables:
            raise ValueError(f"No IPA pronunciation for {token!r}")
        words.append({"text": token, "syllables": syllables})
    if not words or not any(w["syllables"] for w in words):
        raise ValueError("No IPA syllables in line")
    return words
