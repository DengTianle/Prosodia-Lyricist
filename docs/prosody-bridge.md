# Parallel melody → IPA template → XAI lyrics

The standalone bridge predicts XAI-compatible IPA templates using supervised
position queries. The original XAI template-to-lyrics checkpoint is used only
at inference and is not changed by bridge training. New bridge checkpoints use
format v8; older v2-v7 checkpoints retain their original inference architecture.

## Melody encoding

The contrastive melody tower receives its original 177-dimensional note features
in consecutive two-line windows, including the shorter final window. It returns
contextual note vectors before pooled projection. Real windows are packed into
one tower call and scattered back into song order. Padding never creates notes
or windows.

LayerNorm/linear/GELU adapts the vectors to readout width. Global sinusoidal note
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

## Parallel prediction and supervision

Each requested syllable position has a structural query: slot type, line ID,
within-line index, line syllable count, and sinusoidal target position. Source
note positions and target syllable positions are separate coordinates.

Four pre-norm cross-attention/FFN layers retrieve from the complete song memory.
There is no query self-attention, causal mask, previous-label embedding, target
shift, or iterative prediction. All positions are evaluated together. The two
heads predict stress and vowel length independently, with

```text
loss = CE_stress + CE_length
```

Gold labels are loss targets only. Both losses train the readout and, when
unfrozen, the melody encoder. Fixed line prefixes exist only for structural
serialization; they and padding are excluded from the loss.

The default `prosody_mode: actual` uses XAI's `strong/substrong/weak` stress and
`long/short` vowel length. Only the IPA long mark denotes long, matching the
historical dictionary-first pronunciation path. Binary paper-mode data remains
supported for the earlier four-stream lyric decoder, with a two-class stress
head. A checkpoint's vocabulary and IPA rules must match its prepared data and
lyric decoder; secondary stress is not collapsed for XAI.

Reference IPA counts specify query positions during training and evaluation.
MIDI markers specify phrases at inference, and every note supplies one query.
All notes, including an unmarked trailing phrase, are retained. Melisma can
produce note/IPA count differences. There is no learned count head or guaranteed
note-to-syllable alignment. No lyric words, title, or IPA labels enter the melody
source. The title is supplied only to XAI.

## Configuration and workflow

Use the `prosodia-lyricist` Conda environment and `configs/bridge.yaml`; the README
contains preparation, training, evaluation, and MIDI commands.

- `model.bridge_readout` controls query width, layers, heads, FFN width, and dropout.
- `model.song_encoder_layers` controls global source attention (zero bypasses it).
- `model.melody_initialization` selects pretrained or random. Random mode never
  reads the checkpoint, and requires `model.random_melody_encoder`.
- `training.melody_unfreeze_epoch` is zero-based: 1 warms up the new components,
  0 trains the tower immediately, and null freezes it throughout.
- `training.learning_rate` applies to the adapter/song encoder/readout/heads;
  `melody_learning_rate` separately controls the tower.
- Source/window/song and target limits skip entire training examples, with counts
  per reason. Inference rejects overlong input without truncation.
- The existing contrastive-import, song grouping, checksum, and pretraining-split
  audits remain active. Strict audit mode requires the unchanged upstream CSV;
  warning mode records incomplete/conflicting audits while retaining data checks.

XAI training requires repo-mode `data/dali-xai` data. Existing prepared data is
reusable when its manifest configuration matches; old paper-mode labels require
separate preparation. Training uses FP32 or supported CUDA BF16, slot-weighted
gradient accumulation, clipping, and validation-CE early stopping. It never
loads the test split or lyric model. Label-history masking is rejected because
there is no history.

## Evaluation and inference

Evaluation schema v4 contains one `prediction` block. Loss, stress/length/pair
accuracy, exact phrase templates, and phrase-mean unsmoothed BLEU-4 come from the
same parallel pass. Reference IPA counts define the skeleton; count mismatches
with notes remain diagnostics. Duration baselines and constant controls retain
identical coverage, including the equal-count subset. Equal counts do not prove
phonetic alignment.

MIDI inference performs one parallel readout over all song queries. Hard head
predictions are serialized into the unchanged XAI source format, including its
stress tokens, vowel-length IDs, remainder IDs, title, and count prompts.
References never construct the model source. Reports retain source notes and a
separate heuristic comparison. Non-4/4 input and tempo changes are supported.
The lyric model's original generation controls and source/target limits remain
active. Template prediction does not guarantee exact lyric syllable agreement.

Tests cover label independence, both losses and gradients, complete-song context,
padding, accumulation, reload, vocabulary/rule checks, single-pass evaluation,
MIDI tails/tempo/meter, and the XAI handoff including reference isolation.
