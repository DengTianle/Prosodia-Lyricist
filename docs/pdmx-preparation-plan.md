# PDMX preparation audit and plan

Audited 2026-10-30 using the `prosodia-lyricist` conda environment. Recommendation: build a lossless, score-based intermediate dataset, then derive training views from it. Keep melody encoding independent of extraction so the existing 177-D representation and a new score representation can be compared on the same examples.

**Local inventory and scope**

| Stage | Score rows |
| --- | ---: |
| Full CSV / metadata files | 254,077 / 254,077 |
| Available MXL files | 254,035 |
| CSV `has_lyrics` | 42,526 |
| Lyrics + `subset:no_license_conflict` | 28,124 |
| Above, with available MXL | 28,123 |
| Lyrics + no conflict + upstream deduplicated | 23,475 |
| Above, with available MXL | 23,474 |
| Above, passing the preliminary melody/text probe | 14,348 |
| Lyrics + no conflict + deduplicated + rated | 2,357 |

The probe requires at least 20 alphabetic lyric items, 10 whitespace words, and 20 pitched notes in one `(part, staff, voice, verse)`, with no overlapping pitched notes within a measure. It is a diagnostic threshold, **not an English, valid-underlay, or training-ready count**. All 28,123 available no-conflict lyric candidates parsed; the remaining row has `mxl=NA`, rather than a missing download. All CSV metadata/MXL paths with released files exist locally. The CSV MD5 is `30392ccf38bb63ce70e7afae70f9c88c`, matching the [Zenodo v9 release](https://zenodo.org/records/15571083).

Use the CSV's `mxl` and `metadata` columns directly, resolved against the dataset root. `path` identifies the optional MusicRender JSON, which is not downloaded. Metadata basenames are numeric MuseScore IDs; MXL basenames are hashes. Joining them by filename stem would be wrong. Deduplication references remain useful identifiers even when their referenced JSON files are absent.

**Findings that affect preparation**

Counts below concern the 23,474 parsed deduplicated/no-conflict lyric scores; flags overlap.

- **No explicit lyric line or language labels:** zero `<end-line>`, `<end-paragraph>`, or lyric/text `xml:lang` tags. The inspected candidate metadata also has no language keys. Printed system/page breaks are layout information, not DALI-style lyric lines.
- **False-positive lyric flags:** 7,791 scores fail the text probe. Some contain only an attribution such as “By Traditional American” attached as a lyric. The 25th percentile of maximum lyric items in a stream is just one. A negative control of 256 CSV non-lyric scores found zero XML lyric elements; this does not establish full recall.
- **Multiple vocal choices:** 10,941 have lyrics across multiple part/staff/voice combinations; 4,465 have multiple verses in a voice. A piano-named or unnamed part can still carry the sung melody. Instrument name, first part, and highest pitch are insufficient selection rules.
- **Underlay needs validation:** 5,216 have a malformed begin/middle/end word chain in at least one stream; 4,100 contain identical duplicate lyric elements on a note. Diagnose the selected stream rather than rejecting every score with a problematic secondary stream. Literal hyphens and incorrectly marked single syllables can also leave fragmented words without triggering the chain check.
- **Score features matter:** 17,286 contain ties, 6,594 extensions, 3,575 repeats, and 2,425 tuplets. Time signatures occur in 23,114. The longest qualifying melody stream has a median of 129 notes, 95th percentile 428, and maximum 3,366; these are written note counts before tie merging or repeat expansion.
- **English is a separate filter:** in a deterministic sample of 512 deduplicated/no-conflict lyric rows, 129 passed the text threshold and had English as `langdetect`'s top result; 110 also passed the pitch/overlap probe for the selected longest stream. These are automatic estimates, not accepted examples. Latin is unsupported by that detector; mixed language, archaic spelling, solfege, and fragmented words need additional checks. No full English or IPA yield has been measured.

Upstream deduplication preserves different arrangements, rather than isolating works for evaluation. The available deduplicated candidates span 13,264 `best_path` groups, with 3,775 groups containing multiple rows. Candidate metadata also contains 312 MuseScore song IDs that connect multiple `best_path` groups. The [paper's deduplication method](https://arxiv.org/html/2409.10831v2#S3.SS3) is useful evidence, not a complete leakage guarantee.

An exact normalized artist/composer + title/song-name comparison against the current `data/dali-xai/manifest.json` found **141 PDMX candidate scores matching 105 DALI IDs**: 81 train, 12 validation, 12 test IDs. This is identity evidence requiring review, not a full cover/melody audit. Some matches name modern commercial songs, so `no_license_conflict` should be recorded as an upstream metadata flag, not presented as independent verification of the underlying work's rights.

**Recommended data contract**

Keep one canonical record per extracted score voice, with parallel verse underlays. Derive a training example per selected voice/verse without duplicating the stored score. Every derived example retains its work group and source-note references.

| Layer | Preserve |
| --- | --- |
| Provenance | Release DOI, CSV hash, MXL/metadata paths and hashes, score/catalog IDs, license/subset flags, parser and policy versions |
| Identity | Stable namespaced source ID, work group, arrangement group, part/staff/voice and verse IDs |
| Score events | Measure index and printed number, rational onset/duration in quarter-note units, note/rest, written pitch spelling and sounding pitch, tie/attack groups, grace status, tuplets, effective meter/key/tempo/transposition |
| Segments | Ordered event IDs, boundary source and confidence; distinguish score segments from actual lyric lines |
| Underlay | Raw lyric elements, reconstructed words, sung syllables, verse, many-to-many links to original note IDs, unresolved spans and reasons |
| Targets | Existing text-derived IPA/stress/vowel-length labels, pronunciation provenance, word-level count agreement and supervision masks |

Use exact rationals, e.g. `[3, 2]` quarter notes, in canonical data. Do not commit to a fixed quantization grid during extraction. Lyrics and IPA are target/alignment data; the melody encoder receives only score events and the declared source segmentation.

**Extraction and selection policies for the first version**

1. **Index before expensive processing.** Start from all lyric/no-conflict rows, check MXL availability, and freeze identity groups/splits. Retain upstream deduplicated/rated flags for comparison. Select the best usable lyric-bearing arrangement after quality checks: the upstream preferred arrangement may have inferior or absent underlay. Do not require ratings or local MIDI/PDF availability.
2. **Parse the written score faithfully.** Resolve the MXL root through `META-INF/container.xml`. Handle changing divisions, backup/forward, chords, pickups, implicit measures, changing meter, rests, and tuplets. Read pitch alteration and transposition explicitly; retain written and sounding pitch. Keep notated notes and their tie-merged attack mapping. Grace notes remain separately represented; never fabricate positive durations to satisfy the existing encoder. Quarantine unsupported unpitched/microtonal or ambiguous polyphonic cases in the initial training view.
3. **Choose a vocal stream explicitly.** Rank lyric-bearing part/staff/voice candidates by coherent word reconstruction, substantial coverage, monophony, and language evidence. Store alternatives and selection reasons; verify a stratified sample manually. Initially select one principal stream per score, allowing several verses of that stream. Do not concatenate simultaneous choir parts into a single melody. Collapse exact duplicate underlays only when verse, note, and content agree; conflicting duplicates need review.
4. **Reconstruct words and alignment.** Respect `<syllabic>` and formatting fragments; keep verse numbers separate. Preserve elision support even though no `<elision>` occurred in this candidate pool. Merge validated tie chains without dropping source IDs. Treat explicit extensions as melisma evidence; an unlabeled note or a slur alone does not prove continuation. Preserve unresolved instrumental/continuation spans. Block-text verses have no automatic note alignment. The [MusicXML lyric specification](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/lyric/) defines these distinctions. The upstream [reader](https://raw.githubusercontent.com/pnlong/PDMX/main/reading/read_musicxml.py) flattens lyrics into timestamped strings and does not retain all the verse/underlay information needed here.
5. **Keep written and performed order distinct.** Use written order initially, with verses as parallel targets; preserve repeat/navigation structure. Defer ambiguous endings, D.C./D.S., and verse-dependent repeat playback until covered by fixtures. Any expanded view must record measure occurrence and verse routing. Do not combine CSV MusicRender timing/counts with XML note indices: their parsing and repeat conventions can differ.
6. **Declare a segmentation policy.** The corpus lacks DALI's observed line membership. For the main lyric-free task, derive segments from score information such as rests, musical phrase markers, and bar structure using a versioned deterministic policy. Pilot this policy before fixing thresholds. Underlay/punctuation-based segmentation is a separate supplied-skeleton experiment, with its dependency disclosed. Printed systems are never the sole line rule. Flag word/boundary conflicts; never silently join across discarded material. Preserve complete canonical streams even when a training view exceeds limits or fails alignment.
7. **Apply English/IPA checks after reconstruction.** Use whole-verse language evidence, lexical coverage, and manual review of uncertain/mixed cases. Preserve original spelling and documented normalization. Reuse `ipa.parse_words(..., prosody_mode="actual")` and historical pronunciation provenance. IPA stress and vowel length remain text-derived; beat strength and note duration are source features, not substitutes for those targets. Choose pronunciation independently of score syllable counts, and mask/report within-word mismatches under a strict alignment policy.

**Encoding and inference**

A practical first score encoder uses one event per note/rest and combines embeddings for pitch, bar/beat position, rational duration, tie state, and meter; key and notation features can be added as ablations. A REMI-like event-token view is another option. Both should come from the same canonical records. A fixed grid needs measured tuplet/quantization coverage and a versioned vocabulary fitted on training data.

The current `melody_encoder/encoding.py` produces 177-D pitch/duration/IOI features from second-valued arrays. `bridge_data.py` and `bridge_pretrain.py` hard-code that interface, and checkpoints identify the representation. A score encoder therefore needs an explicit representation selector, matching encoder/checkpoint metadata, and revised masked-event objectives if pretraining is used. Optional contrastive fusion can stay disabled. An old-representation comparison may use a declared tempo-derived timing adapter; beats must not simply be relabeled as seconds or treated as observed performance timing.

The current `bridge_model.generate()` defaults to one slot per note, and MIDI inference uses that behavior. Score melismas require an additional decision: either supply an underlay/slot map explicitly, or predict syllable starts/continuations from lyric-free score input. The initial baseline can use one slot per tie-merged attack, with that limitation measured. Evaluate supplied-underlay and lyric-free settings separately. Exact IPA alignment from training lyrics must not become an undeclared inference input. Elisions with several syllables on one note also exceed the current single-pair-per-note target design.

**Split and implementation plan**

Build transitive work groups from `best_path`, catalog song IDs, trustworthy normalized composer/title identities, and corroborated melody/long-text fingerprints. Keep all arrangements, voices, verses, repeats, and augmentations of a work together. Avoid merging unrelated works solely on generic titles or short common lyrics. Freeze approximately 80/10/10 groups with the existing seeded hash utility before IPA rejection.

When transferring to or jointly evaluating DALI, protect its existing held-out works: matched validation/test works cannot enter PDMX training or masked pretraining. Pin compatible work groups to the existing split or exclude them from the training pool; quarantine conflicting assignments. Audit upstream encoder and lyric-decoder training exposure too. Missing identity matches do not establish absence of overlap.

Suggested implementation order:

1. Add a score reader/canonicalizer (`musicxml.py`) and an independent entry point (`prepare_pdmx.py`); leave DALI-specific annotation parsing in `dali.py`.
2. Produce a roughly 100-score stratified pilot spanning simple English melody, SATB, multiple verses, ties/melismas, tuplets/pickups, repeats, bad underlay, and attribution-only false positives. Review selected voices, words, and boundaries against rendered scores before scaling.
3. Add source-specific alignment dispatch to `bridge_alignment.py`; the current `dali_word_note_starts_v1` scheme must not be reused for XML provenance. Preserve strict IPA/sung-count mismatch masks and full note-ID mappings.
4. Add a versioned PDMX manifest/data contract in `data.py`, plus the selected score encoding in the bridge dataset/model/checkpoint paths. Keep representation changes independent of IPA target rules. Retain the existing 40-syllable/2,048-note limits for a comparable pilot, and count every exclusion rather than truncating.
5. Materialize separate `data/pdmx-prepared/{train,valid,test}.jsonl`, `manifest.json`, `rejected.jsonl`, and split/quality reports. Publish staged outputs with the manifest last, as the DALI preparer does. Report work/score/voice/verse/segment counts, language uncertainty, alignment coverage, and exclusions separately. Balance repeated verses/choir duplicates so they do not dominate training.

Acceptance checks should cover changing divisions and voices, ties across bars, melismas versus instrumental gaps, verse duplication, word reconstruction, pickups/tuplets, repeat routing, and source/target independence. Check deterministic output hashes, disjoint work groups including DALI exposure, and consistent note references. Strip or shuffle lyric/IPA target fields and verify source features stay identical under the declared score-only segmentation. Measure actual English/IPA yield and note/segment limits on the pilot before a full preparation run.

**Downloads and audit artifacts**

No additional Zenodo files are required for the recommended MusicXML preparation. `data.tar.gz` (2.2 GB) is optional for MusicRender cross-checks; MIDI is unnecessary, subset path lists duplicate CSV flags, and the full PDF archive (9.6 GB) is unnecessary for a small rendering review. The MXL files were exported with MuseScore separately from MusicRender JSON generation in the [release builder](https://raw.githubusercontent.com/pnlong/PDMX/main/wrangling/pdmx.py).

Machine-readable results and the local audit scripts are under `data/pdmx-audit/` (already ignored by Git): `summary.json`, `scores.jsonl`, `language_estimates.json`, `negative_control.json`, and `dali_identity_overlap.json`. The audit deliberately stops before full language classification, MusicXML schema validation, complete cross-measure timing/underlay validation, IPA extraction, or training.

Reproduce the main structural audit and sample language estimates with:

```bash
conda run --no-capture-output -n prosodia-lyricist python data/pdmx-audit/audit_pdmx.py
conda run --no-capture-output -n prosodia-lyricist python data/pdmx-audit/analyze_language_sample.py
```
