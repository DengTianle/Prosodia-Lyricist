# Baseline MIDI inference and evaluation

## Run the Imagine sanity check

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.infer \
  --checkpoint checkpoints/dali/20260913T125414942262Z/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --output outputs/imagine-baseline.txt \
  --report-prefix outputs/imagine-baseline
```

The checkpoint path above is the locally available trained baseline, not a
downloadable artifact. For another run, substitute its `best/` directory.
Output paths must be new: choose another prefix to preserve previous results.
`--top-k 1` is greedy; sampling uses `--top-k N --temperature T --seed S`.

The UTF-8 `.txt` contains generated lyrics. The Markdown report contains the
input and output templates, words, IPA, note pitches/ticks, metrical positions,
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

No checkpoint weights, tokenizer IDs, DALI preparation rules, or training
targets change. In particular, the historical baseline IPA training labels
retain separate secondary stress and do not classify diphthongs as long.
The evaluation extractor below uses the paper's binary rules independently.

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
  --checkpoint checkpoints/dali/20260913T125414942262Z/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --reference /path/to/reference.txt --report-prefix outputs/imagine-reference
```

Supply exactly one nonempty lyric line per input phrase, without a title/header.
The Markdown report shows these ground-truth lyrics and their IPA-derived
prosody beside the MIDI and generated labels, using the same evaluation rules
as generated lyrics. The JSON report also stores `reference_syllables`.
Target construction matches training's wordwise byte-BPE tokenization with
period boundaries and one BOS/EOS pair. The text is only a scoring target and
never enters the source or generation call. Overlength references fail without
truncation. No reference lyrics are bundled or inferred from the title.

The public BART inference code returns placeholder `ppl = 0.0`; the paper does
not pin enough evaluator/tokenization details to reproduce its numeric PPL.
This implementation provides a reproducible, explicitly named conditional BPE
metric. It is not an external language-model score. Reuse identical conventions
when adding evaluation to `template-decoder`, and exclude its auxiliary losses.

## Local verification

Tests cover the supplement figure's patterns, rounding, 4/4 validation, global
versus phrase means, Imagine's unmarked tail, compound-symbol BLEU, clipping,
brevity, missing/extra/failed phrases, IPA rules, NLL masking, token-weighted
aggregation, causal teacher forcing, report export, and template-only use.
Run `python -m pytest -q` in the `prosodia-lyricist` conda environment.
