# Note heads → IPA template → XAI lyrics

The standalone bridge predicts XAI-compatible IPA templates using supervised
note classification. The original XAI template-to-lyrics checkpoint is used only
at inference and is not changed by bridge training. New bridge checkpoints use
format v9; older v2-v8 checkpoints retain their original inference architecture.

## Melody encoding

The contrastive melody tower receives its original 177-dimensional note features
in consecutive two-line windows, including the shorter final window. It returns
contextual note vectors before pooled projection. Real windows are packed into
one tower call and scattered back into song order. Padding never creates notes
or windows.

LayerNorm/linear/GELU adapts the vectors to song-encoder width. Global sinusoidal note
positions and shared song-line embeddings are added after the tower, preserving
its pretrained feature normalization.

A separate 39-dimensional channel retains whole-song timing/pitch relationships.
Its seven scalars are pitch from the song's first note and preceding-note pitch
interval (in octaves), log2(duration/scale), log2(positive IOI/scale), previous-note
and positive-IOI indicators, and log2(scale in seconds). Scale is the median
positive song IOI, falling back to median duration. Missing/zero IOIs have zero
log value and explicit indicators. The other 32 features are sine/cosine time
coordinates for periods 2^-1 through 2^14 in scale units from the first onset.
This is not an inferred beat grid. Window-boundary intervals are preserved.

A linear/GELU/linear/LayerNorm projection adds these features to the adapted
vectors. After fusion normalization, two bidirectional song-encoder layers
attend across the complete song.

## Note prediction and supervision

Dropout and two shared linear classifiers read each note's contextual vector
directly after the song encoder. There are no target queries, count/slot
embeddings, cross-attention readout, or label-history inputs in this architecture.
Stress and vowel length are predicted independently at all notes, with

```text
loss = CE_stress + CE_length
```

Observed DALI word membership and sung-group starts select the first note of
each sung syllable. Its predictions are gathered into the corresponding IPA
slot. Counts, note anchors, and labels affect supervision only; they never
condition the encoder or heads. Fixed line prefixes and padding have no loss.
Both losses train the song encoder and, when unfrozen, the melody tower.

`alignment_policy: strict` masks a word if its sung-group and IPA counts differ.
`prefix` pairs their first `min(sung, IPA)` positions and masks the rest, always
restarting at the next word. Spelling/token boundaries are matched within raw
words; unresolved text is masked. All notes remain context. Examples with no
supervised slots are skipped and counted. Continuation-note outputs have no
direct CE target; there is no learned role head.

The default `prosody_mode: actual` uses XAI's `strong/substrong/weak` stress and
`long/short` vowel length. Only the IPA long mark denotes long, matching the
historical dictionary-first pronunciation path. Binary paper-mode bridges remain loadable for template evaluation;
the current XAI handoff requires the three-class repo vocabulary. A checkpoint's vocabulary and IPA rules must match its prepared data and
lyric decoder; secondary stress is not collapsed for XAI.

Reference IPA counts specify the scoring skeleton during training and evaluation.
MIDI markers specify phrases at inference, and every note supplies one output pair.
All notes, including an unmarked trailing phrase, are retained. Melisma can
produce note/IPA count differences. There is no learned count head. No lyric
words, title, or IPA labels enter the melody
source. The title is supplied only to XAI.

## Configuration and workflow

Use the `prosodia-lyricist` Conda environment and `configs/bridge.yaml`; the README
contains preparation, training, evaluation, and MIDI commands.

- `model.query_mode: note` selects direct note heads.
- `model.song_encoder` controls width, attention heads, FFN width, and dropout.
- `model.alignment_policy` selects `strict` (default) or word-local `prefix`.
- `model.song_encoder_layers` controls global source attention (zero bypasses it).
- `model.melody_initialization` selects pretrained or random. Random mode never
  reads the checkpoint, and requires `model.random_melody_encoder`.
- `training.melody_unfreeze_epoch` is zero-based: 1 warms up the new components,
  0 trains the tower immediately, and null freezes it throughout.
- `training.learning_rate` applies to the adapter/song encoder/heads;
  `melody_learning_rate` separately controls the tower.
- `training.length_class_weights` is null (ordinary CE) or `[long, short]`, e.g.
  `[2.0, 1.0]`. Only length CE is weighted. `loss`/`components` always report
  ordinary CE; `objective_loss`/`objective_components` report the weighted
  objective when enabled. Validation uses the same weights for checkpoint
  selection. Gradient accumulation normalizes each head over the complete group.
- Source/window/song and target limits skip entire training examples, with counts
  per reason. Inference rejects overlong input without truncation.
- The existing contrastive-import, song grouping, checksum, and pretraining-split
  audits remain active. Strict audit mode requires the unchanged upstream CSV;
  warning mode records incomplete/conflicting audits while retaining data checks.

XAI training requires repo-mode `data/dali-xai` data. Existing prepared data is
reusable when its manifest configuration matches. Schema v6 stores note anchors;
older files recover them in memory from raw DALI without rerunning pronunciation.
Recovery verifies note pitch/timing and records raw file checksums. The evaluator's
`--dali-dir` overrides old cluster paths. Old paper-mode labels require separate
preparation. Training uses FP32 or supported CUDA BF16, supervised-slot-weighted
gradient accumulation, clipping, and validation-CE early stopping. It never
loads the test split or lyric model. Label-history masking is rejected because
there is no history.

## Evaluation and inference

Evaluation schema v5 contains one `prediction` block. Loss, stress/length/pair
accuracy, exact phrase templates, and phrase-mean unsmoothed BLEU-4 come from the
same encoder pass. Slot metrics exclude masked targets. Exact-template accuracy
and BLEU score only fully supervised phrases; both coverage counts are reported.
Reference IPA counts define the skeleton; count mismatches remain diagnostics.
Duration baselines use identical note anchors and masks, including the equal-count
subset. Equal counts do not prove phonetic alignment.

MIDI inference classifies every note once. Hard head
predictions are serialized into the unchanged XAI source format, including its
stress tokens, vowel-length IDs, remainder IDs, title, and count prompts.
References never construct the model source. Reports retain source notes and a
separate heuristic comparison. Non-4/4 input and tempo changes are supported.
The lyric model's original generation controls and source/target limits remain
active. Template prediction does not guarantee exact lyric syllable agreement.

Tests cover label independence, both losses and gradients, complete-song context,
padding, accumulation, reload, vocabulary/rule checks, single-pass evaluation,
MIDI tails/tempo/meter, and the XAI handoff including reference isolation.
