"""Versioned repo/paper IPA rules and historical dictionary-first pronunciation."""

from functools import lru_cache
from importlib.resources import files

MODERN_POLICY = "modern_prosodic_first_wordform"
HISTORICAL_POLICY = "prosodic_1_5_dictionary_first"
HISTORICAL_SOURCE = "prosodic_1_5_dictionary"
FALLBACK_SOURCE = "modern_prosodic_fallback"
PROSODY_MODES = ("paper", "actual")
PAPER_RULES = "ipa_binary_stress_diphthong_length_v2"
REPO_RULES = "ipa_repo_secondary_longmark_historical_v1"
REPO_STRESSES = ("strong", "substrong", "weak")


def _resources():
    return files("prosodia_lyricist").joinpath("resources", "prosodic_1_5")


def historical_provenance():
    from .historical_ipa import verify_resources

    return verify_resources(_resources())


@lru_cache(maxsize=1)
def _historical_lexicon():
    historical_provenance()
    resources = _resources()
    first = {}
    for line in resources.joinpath("english.tsv").read_text(encoding="utf-8").splitlines():
        token, value, *_ = line.split("\t")
        token, value = token.strip(), value.strip()
        if token.startswith("#"):
            continue
        if "." in token and token.count(".") == value.count("."):
            token = token.replace(".", "")
        first.setdefault(token, value)
    ambiguous = frozenset(resources.joinpath("maybestressed.txt").read_text().split())
    unstressed = frozenset(resources.joinpath("unstressed.txt").read_text().split())
    return first, ambiguous, unstressed


def historical_syllables(token):
    """v1.5 dictionary order and maybeUnstress precedence; no melody-dependent selection."""
    first, ambiguous, unstressed = _historical_lexicon()
    token = token.lower()
    value = first.get(token)
    if value is None:
        return None
    force_weak = token not in ambiguous and token in unstressed
    return [
        repo_syllable_features(s.translate(str.maketrans("", "", "'`ˈˌ")) if force_weak else s)
        for s in value.split(".")
    ]


def repo_syllable_features(syllable):
    value = syllable if isinstance(syllable, str) else syllable.ipa
    return {
        "ipa": value,
        "stress": (
            "strong"
            if any(mark in value for mark in ("'", "ˈ"))
            else "substrong"
            if any(mark in value for mark in ("`", "ˌ"))
            else "weak"
        ),
        "length": "long" if "ː" in value else "short",
    }


# English IPA diphthongs, including common British and American transcriptions.
DIPHTHONGS = ("eɪ", "aɪ", "ɔɪ", "əʊ", "oʊ", "aʊ", "ɪə", "eə", "ɛə", "ʊə")


def syllable_features(syllable):
    # Modern Syllable.__str__ displays orthography, not IPA.
    ipa = syllable if isinstance(syllable, str) else syllable.ipa
    return {
        "ipa": ipa,
        "stress": "strong" if any(mark in ipa for mark in ("'", "ˈ", "`", "ˌ", '"')) else "weak",
        "length": "long"
        if "ː" in ipa or ":" in ipa or any(d in ipa for d in DIPHTHONGS)
        else "short",
    }


@lru_cache(maxsize=1)
def backend():
    try:
        import prosodic
    except ImportError as exc:
        raise RuntimeError(
            "IPA preparation/correction requires modern prosodic; install the ipa extra"
        ) from exc
    try:
        # Materialize lazy words too, so missing resources fail before output creation.
        list(prosodic.Text("hello", lang="en", syntax=False).wordtokens)
    except (AttributeError, LookupError) as exc:
        raise RuntimeError(
            "IPA preparation/correction requires prosodic>=3.10 and its pronunciation/tokenizer "
            "resources. Install the ipa extra and NLTK punkt/punkt_tab data."
        ) from exc
    return prosodic


def parse_words(text, *, prosody_mode="paper", pronunciation_policy=None):
    """Select the first pronunciation per word, independently of DALI sung counts."""
    if prosody_mode not in PROSODY_MODES:
        raise ValueError(f"Unknown prosody_mode: {prosody_mode}")
    policy = pronunciation_policy or (
        HISTORICAL_POLICY if prosody_mode == "actual" else MODERN_POLICY
    )
    if policy not in (MODERN_POLICY, HISTORICAL_POLICY):
        raise ValueError(f"Unknown pronunciation policy: {policy}")
    try:
        return _parse_words(text, prosody_mode=prosody_mode, pronunciation_policy=policy)
    except (OSError, LookupError) as exc:
        # Environment failures must abort preparation, not mark songs as bad data.
        raise RuntimeError(f"IPA backend resource/cache failure: {exc}") from exc


def _parse_words(text, *, prosody_mode, pronunciation_policy):
    text = text.translate(str.maketrans({quote: "'" for quote in "‘’ʼ‛‚′‵ꞌʹʻ"}))
    parsed = backend().Text(
        text.replace("cuz", "cause").replace("-", " ").replace(".", ""),
        lang="en",
        syntax=False,
    )
    words = []
    for word in parsed.wordtokens:
        token = word.txt.strip()
        if not token:
            continue
        historical = pronunciation_policy == HISTORICAL_POLICY
        syllables = historical_syllables(token) if historical else None
        source = HISTORICAL_SOURCE
        if syllables is None:
            form = word.wordtype.form
            features = repo_syllable_features if prosody_mode == "actual" else syllable_features
            syllables = [features(s) for s in form.syllables] if form is not None else []
            source = FALLBACK_SOURCE
        # Punctuation tokens (including '?') have no pronunciation by design.
        # Retain them with zero syllables, just like commas and exclamation marks.
        if any(c.isalpha() for c in token) and not syllables:
            raise ValueError(f"No IPA pronunciation for {token!r}")
        entry = {"text": token, "syllables": syllables}
        if historical and syllables:
            entry["pronunciation_source"] = source
        words.append(entry)
    if not words or not any(w["syllables"] for w in words):
        raise ValueError("No IPA syllables in line")
    return words
