# Template-decoder MIDI inference and evaluation

## Run the Imagine sanity check

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.infer \
  --checkpoint checkpoints/dali/TEMPLATE_RUN/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --output outputs/imagine-template-decoder.txt \
  --report-prefix outputs/imagine-template-decoder
```

No trained template-decoder checkpoint is available yet. Replace `TEMPLATE_RUN`
with a newly trained explainable run's `best/` directory when ready. For now, use
`--template-only` below. Tiny-model tests validate the full pipeline, not lyric
quality. Historical baseline checkpoints still load through the legacy path;
reports explicitly identify `decoder_mode` so they cannot be mistaken for the
explainable model.
Output paths must be new: choose another prefix to preserve previous results.
`--top-k 1` is greedy; sampling uses `--top-k N --temperature T --seed S`.

The UTF-8 `.txt` contains generated lyrics. The Markdown report contains the
input and output templates, words, IPA, note pitches/ticks, metrical positions,
per-word predicted versus corrected decoder labels,
phrase counts, per-phrase BLEU, and perplexity. The JSON preserves all of this,
raw generated text/token IDs, the exact encoded source, MIDI checksum, package
versions, checkpoint run metadata, and generation settings. The positional
tables are inspection aids, not a computed syllable-to-note alignment.

Inspect the MIDI without loading a checkpoint or querying IPA:

```bash
python -m prosodia_lyricist.infer --midi examples/imagine.mid --title Imagine \
  --template-only --report-prefix outputs/imagine-template
```

## Unit and phrase boundaries

**One complete MIDI melody is one model input and one generation call.** All
phrase templates share the title and encoder/decoder context. There is no
paragraph batching, independent line generation, automatic chunking, or silent
truncation. Source and target limits come from the checkpoint. A shorter,
phrase-marked MIDI can be supplied as a shorter song, but this changes context.

Period-delimited generated text becomes the output lyric lines. The decoder
does not enforce a line count or syllable count. The report warns if line
counts differ or the generation token budget is reached. A final forced EOS
at the budget does not prove that the song was completed naturally.

As in the released XAI inference code, each marker ends a phrase and includes
notes whose **onsets equal the marker time**. Do not insert bar/beat markers
into that marker stream. Notes following the final marker become an additional
phrase rather than being discarded. Empty marker intervals are ignored.

The supplied Imagine file has PPQ 480, explicit 4/4 at tick zero, 113 notes,
16 markers, and a 10-note unmarked tail. It therefore produces **17 phrases**,
with counts `[7, 6, 5, 5, 9, 7, 7, 6, 5, 5, 9, 5, 5, 7, 7, 8, 10]`.
The first phrase has 7 notes, including the one beginning at its marker (1800).

## MIDI-to-template standard

Sources: XAI-Lyricist supplementary Section 1.1, Eqs. (1)-(3), Figure 1;
`docs/references/Supplementary Materials.pdf`. The adjacent reference checkout's
`utils/prosody_utils.py::stress` is used to resolve note-type cases the figure
does not illustrate. The implementation is versioned as a documented paper
reconstruction, not a claim of bit-for-bit identity to the authors' experiment.

Default `--stress-source supplement`:

- Only 4/4 is supported. Reject explicit other meters, including meter changes
  to non-4/4. Missing meter assumes 4/4 and is flagged. MIDI tick zero establishes
  the grid; pickups retain their actual positions and phrases never reset it.
- One monophonic, non-drum track (default index 0), one syllable per note.
  Melisma is not inferred. Tempo does not change tick-based relative features.
- Strength depends on both onset and note type, as in Figure 1. For ordinary
  durations `PPQ * {1/16, 1/8, 1/4, 1/2, 1, 2, 4}`, set the rhythmic grid unit
  to the duration; otherwise use a quarter note (`PPQ`), following the released
  helper's fallback for dotted/irregular notes. Quantize onset to the nearest
  grid position using `floor(onset / unit + 0.5)` (ties upward). Positions
  0 and 2 modulo 4 are strong; 1 and 3 are weak. Thus quarter, eighth, and
  sixteenth note sequences reproduce Figure 1 with sub-strong mapped to strong.
- Length is long **strictly above the mean duration of all selected-track
  notes in this MIDI**, otherwise short (supplement Eq. (3)). The same threshold
  is retained across phrase boundaries. For Imagine it is 313.274336 ticks.

The supplement's Eq. (1) omits an integer/rounding convention for measure number
and appears to omit the conversion to one-based beat numbering used in Eq. (2).
Read as real-valued division, it would make every residual zero. Figure 1 and
the released helper establish the intended duration-dependent alternating
pattern. Nearest-grid half-up quantization and the irregular-duration fallback
above are explicit operational choices; the PDF alone does not uniquely specify
them. We do not silently claim an exact formula reproduction.

The main paper describes phrase-mean length, the supplement says melody mean,
and the public `getProsody` implementation uses onset buckets of `8 * PPQ`.
This mode follows the **supplement's melody mean**, as requested. Historical
`--stress-source heuristic` retains this project's previous unquantized
duration-dependent strength plus phrase-mean length. `unknown` also preserves
the historical phrase mean and disables strength. Unknown-trained checkpoints
automatically select `unknown`; explicitly pass `supplement` to override.

This evaluation port changes no training targets, tokenizer IDs, or checkpoint
weights. Template-decoder's existing training rules already use binary stress
and diphthong length. Evaluation still re-extracts prosody independently from
the final text, never trusting the decoder's prediction heads as its own score.

## Prosody-BLEU

Convert every generated syllable to an atomic `(strength, length)` symbol.
Use modern Prosodic's first wordform; primary **or secondary** IPA stress means
strong, otherwise weak. A length mark (`ː` or `:`) or an English diphthong means
long. The explicit diphthong inventory is in `evaluation.py`. Original word
text and IPA are saved so pronunciation choices can be inspected.

Each MIDI phrase's template is the single reference; its corresponding generated
phrase is the hypothesis. BLEU-4 uses clipped 1-, 2-, 3-, and 4-gram precisions,
equal weights, and the usual brevity penalty:

`BLEU = exp(min(0, 1 - reference_length / hypothesis_length)
            + 0.25 * sum(log(p_n), n=1..4))`.

No BART tokenization, special tokens, cross-phrase n-grams, smoothing, or
effective-order adjustment enter BLEU. If any required precision is zero,
the score is exactly zero (rather than a floating-point near-zero substitute).
Thus even an identical phrase shorter than four syllables scores zero. Scores
are on the 0..1 scale. Report per-phrase scores and their arithmetic mean.

Pair phrases in order, with a denominator of `max(input_count, output_count)`.
Missing/extra phrases and failed IPA extractions contribute zero and are
reported. Failed IPA is not represented as a known zero syllable count.
Unknown input stresses make BLEU unavailable. Resource/installation errors
abort rather than masquerading as poor model outputs. If phrase counts differ,
ordinal pairing may misalign later lines; the report explicitly warns about it.

The paper defines similarity of joint prosody patterns but does not fully pin
BLEU configuration. Fixed unsmoothed BLEU-4 follows the default NLTK call in
the released script. That script actually scores stress-only tokenizer IDs,
includes special tokens, and its active non-greedy regex can extract an empty
template. This implementation follows the **joint prosody definition in the
paper**, not those bugs. It should not be compared numerically to Table 2 as an
exact replication. `METRIC_VERSION` records these choices for later branches.

## Perplexity

Conditional BART perplexity is `exp(sum(target negative log probabilities) /
number of scored target BPE tokens)`. Teacher-force the complete sequence with
the exact MIDI-derived source. Use raw model logits, before top-k, temperature,
or generation constraints. BART shifts targets causally. Ignore BOS, PAD and
`-100`; include lyric subwords, punctuation/periods and EOS. Exclude the extra
decoder-start token. Auxiliary losses never enter this calculation. Aggregate
multiple examples via summed NLL and token counts, not the mean of batch PPLs.

The report always includes **generated-sequence conditional perplexity**.
This measures the generator's confidence in its own output, and is not an
independent fluency measure or a held-out reference evaluation. Greedy decoding
can obtain low self-perplexity while producing poor lyrics.

To additionally score reference lyrics under the same MIDI source:

```bash
python -m prosodia_lyricist.infer \
  --checkpoint checkpoints/dali/TEMPLATE_RUN/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --reference /path/to/reference.txt --report-prefix outputs/imagine-reference
```

Supply exactly one nonempty lyric line per input phrase, without a title/header.
`--reference` requires the path to that text file. The Markdown report shows
the actual lyrics, their IPA-derived syllable counts and stress/length labels,
and comparison columns beside the MIDI and generated output. Reference labels
use the same evaluation rules as generated lyrics and are also saved as
`reference_syllables` in the JSON report.
Target construction matches training's wordwise byte-BPE tokenization with
explicit `<word_end>` events, three aligned IPA feature streams, period
boundaries and one BOS/EOS pair. The text is only a scoring target and
never enters the source or generation call. Overlength references fail without
truncation. No reference lyrics are bundled or inferred from the title.

The public BART inference code returns placeholder `ppl = 0.0`; the paper does
not pin enough evaluator/tokenization details to reproduce its numeric PPL.
This implementation provides a reproducible, explicitly named conditional BPE
metric. It is not an external language-model score. The compound-decoder
extension below preserves the baseline normalization where applicable and
excludes all auxiliary losses.

## Local verification

Tests cover the supplement figure's patterns, rounding, 4/4 validation, global
versus phrase means, Imagine's unmarked tail, compound-symbol BLEU, clipping,
brevity, missing/extra/failed phrases, IPA rules, NLL masking, token-weighted
aggregation, causal teacher forcing, report export, and template-only use.
Run `python -m pytest -q` in the `prosodia-lyricist` conda environment.


## Compound decoder perplexity

An explainable checkpoint consumes four decoder streams. For generated text,
perplexity replays the actual generated token and feedback streams (corrected
IPA labels by default, sampled labels with `--no-prosody-correction`). Each
stream is shifted by one event; the current label cannot condition its own
prediction. Reference scoring constructs the same four training target streams
from reference IPA. Auxiliary-head losses never enter either score.

The main `perplexity` excludes `<word_end>` events from its NLL and denominator,
but keeps those events and their features in causal decoder context. The
nested `event_perplexity` includes their log probabilities and reports its own
count. Both ignore BOS/PAD and include punctuation and EOS. This is a conditional
lyric-BPE diagnostic given the extra structural history, not a marginal
plain-text probability directly interchangeable with baseline PPL. For an
independent cross-system fluency comparison, use a shared external evaluator
or human judgments. Shared prosody-BLEU remains identical across branches.

A truncated final BPE word is removed from all four streams. The report retains
the event-budget warning and discarded fragment even though the stored sequence
is now shorter than the budget. If only BOS remains, perplexity is unavailable
with zero scored tokens. An empty/missing phrase still contributes zero BLEU.

`--explanations path.json` remains available, including alongside `--report-prefix`.
Reports embed the same explanation object. `--no-prosody-correction` disables
IPA feedback during generation; reports still use independent IPA extraction
for output evaluation. BLEU reports now show the clipped matches and candidate
counts for all four n-gram orders so zero scores can be inspected directly.

## Interpreting the existing baseline sanity check

`imagine-baseline-verified` was the name of a second locally checked output file,
not a distinct checkpoint, split, or human quality certification. Its checkpoint
was `checkpoints/dali/20260913T125414942262Z/best`; generation was greedy
(`top_k=1`, seed 1234, MPS), with the supplement template method. The saved run
metadata identify a DALI-trained, lyrics-only BART-base model, not the original
XAI pretrained artifact or a template-decoder model.

In that run, 12 of 15 generated phrases had no matching compound 4-gram, although
all 12 had unigram matches. Fixed unsmoothed BLEU-4 therefore gives zero. Two
more input phrases have no corresponding output, yielding 14 zeros among the
17 phrase slots. These zeros do not imply no prosodic similarity. The report
pairs output phrases in order: we cannot know from that pairing whether the
model omitted the final phrases or merged/shifted earlier phrases. The decoder
terminated with EOS before its 1023-token budget; this was not length truncation.

The stored reports under the adjacent XAI checkout's `gen_lyrics_*` directories
are not a controlled comparison. For example, `gen_lyrics_20241030:162809` has
16 input phrases and its fifth input phrase has 7 syllables versus 9 notes in
this MIDI. All 16 saved input stress patterns match the saved reference IPA
patterns exactly. That strongly suggests reference-lyric-derived conditioning,
rather than the current MIDI-derived templates, although these output files do
not establish complete historical execution provenance. Its filename also
records temperature 1.3/top-k 3, rather than the current greedy decoding.

The quality gap is real as an observation but its causes are not isolated:
checkpoint/training corpus, conditioning, sampling, and phrase coverage differ.
Baseline training uses lyric-derived IPA while this inference test uses musical
features; that known conditioning shift is one plausible contributor. The
older filename BLEUs near 1e-231 are also not the paper's 0.92/0.98 figures;
the broken reference-script extraction makes those numbers uninformative.
Our reported joint unsmoothed metric is reproducible across these branches,
but has not been numerically validated against the paper's original evaluator.
