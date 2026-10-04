# Contrastive melody → IPA template → lyrics

The v5 bridge models complete songs with separate strength/length heads. It trains
on IPA prosody slots and uses one slot per MIDI note at inference. v2/v3/v4
checkpoints retain their original window-level architectures; v1 requires retraining.

The bridge replaces the heuristic MIDI-to-template step. It reuses the trained
melody trunk and 177-dimensional note encoding from `prosodia-direct`, learns
to predict the template decoder's source labels, and leaves the existing
four-stream lyric checkpoint intact. No HuBERT/audio model is loaded.

## Supervision and model

Each training example is a complete song of consecutive DALI phrases. Melody inputs
contain pitch, onset, duration, and phrase membership. Targets are the ordered
IPA stress/length pairs in each phrase. Words, lyric text, IPA labels and their
counts never enter the melody source. The title is supplied only to the existing
lyric model at inference.

The melody Transformer returns contextual **note vectors before pooled
contrastive projection**. It still encodes two-line windows independently, including
the final shorter window. All real windows in a batch are packed for the tower and
their outputs scattered into song order. The LayerNorm/linear/GELU adapter maps
each note into bridge width 256. Global sinusoidal note positions and learned
song-line embeddings are added after the tower. Padding never occupies a global
note position or an encoder window.

A separate 39-dimensional source channel is computed from raw notes over the
entire song. Its seven scalar features are pitch from the song's first note and
preceding-note pitch interval (both in octaves), log2(duration/scale),
log2(positive IOI/scale), previous-note and positive-IOI indicators, and
log2(scale in seconds). The scale is the median positive song IOI, falling back
to median note duration when none exists. Missing/zero IOIs have log value zero
and explicit indicators. The other 32 features are sine/cosine time coordinates
for periods 2^-1 through 2^14 in scale units, measured from the song's first onset.
This is a common timing reference, not an inferred beat or meter. Pitch/timing
relationships across window boundaries survive; the original window-normalized
177-dimensional pretrained inputs remain unchanged.

A Linear/GELU/Linear/LayerNorm projection adds this channel to the note vectors.
After a fusion LayerNorm, two pre-norm bidirectional song-encoder layers attend
across the entire song. A four-layer causal decoder cross-attends to that memory.
Both stacks use four heads, FFN width 1024, and dropout 0.1. Decoder inputs combine
previous predictions, global target positions, shared song-line embeddings, and
within-line slot/count embeddings. Source note and target token positions are
separate coordinates. Set `song_encoder_layers: 0` for concatenation-only ablation.

The feedback/serialization vocabulary is:

```text
PAD, BOS, SLOT, strong_long, strong_short, weak_long, weak_short,
line_0, line_1, ... line_255
```

For example, two phrases with three and one syllables become:

```text
line_0 weak_short strong_long strong_short line_1 weak_long
```

Training shifts targets right behind BOS and sums separate binary stress/length
cross-entropies at syllable slots. Fixed line prefixes and padding are excluded
from the loss. There is no count head or end-of-line prediction. These are
**source syllable labels**, distinct
from the existing lyric decoder's word-level explanation heads.

`training.history_mask_probability` independently replaces previous prosody-pair
inputs with the existing `SLOT` token during training (0.3 in the example config;
0 or omission disables it). Gold loss targets, BOS/line prefixes, padding, and
slot/count embeddings are preserved. Training loss and accuracies reflect the
masked history; `metrics.jsonl` records the probability. Validation, generation,
checkpoint selection, and the checkpoint format are unchanged.

All notes, including melisma, are retained. There is no forced note-to-syllable
alignment. Encoder windows never cross songs or data splits. Both bridge and
lyric generation operate on complete songs. The bridge retains causal prediction
history across window boundaries and caches source/target attention K/V at inference.

## Configure and train

Use the `prosodia-lyricist` Conda environment and `configs/bridge.yaml`.
Set these paths before preparing:

| Setting | Meaning |
| --- | --- |
| `model.melody_checkpoint` | Original contrastive checkpoint with `mlm_note_177d_v1` and `audio_pooling: note`. This is not a fine-tuned direct-lyrics checkpoint. |
| `data.pretraining_manifest` | Immutable CSV actually used for that contrastive run. |
| `data.lines_per_window` | Prepared-data provenance: pretraining window line count, default 2. |
| `model.encoder_lines_per_window` | Must match the prepared/pretraining count, independently of song scope. |
| `model.bridge_scope` | `song` by default for new training; `window` retains the legacy architecture. |
| `data.prepared_dir` | Separate bridge dataset, default `data/dali-bridge`. |
| `model.max_window_notes` | Maximum notes per pretrained window, default 512. |
| `model.max_song_notes`, `model.max_song_lines` | Source limits, default 2048 notes / 256 lines. |
| `model.max_target_length` | Total template length including prefixes, default config 4096. |
| `model.song_encoder_layers` | Two by default; zero bypasses global source attention. |
| `data.max_syllables` | Maximum IPA syllables per phrase; must fit the lyric checkpoint at inference. |
| `training.melody_unfreeze_epoch` | Zero-based epoch to fine-tune the trunk; 1 warms up the new decoder for one epoch, 0 starts immediately, null freezes throughout. |
| `training.learning_rate` | Adapter, feature projection, song encoder and template decoder rate. |
| `training.melody_learning_rate` | Separate, lower rate for the pretrained trunk. |
| `training.history_mask_probability` | Probability in [0, 1] of masking each previous prosody pair during training; default 0. |

The vendored encoder and strict importer are taken from `prosodia-direct`,
whose upstream integration pins `try-contrastive` revision
`ff6247f2613fd5e1d1f06c44b73f9dadfa6cb257`. CLS and mean pooling checkpoints are
supported; both supply contextual note vectors. The pooled projection weights
are validated on import and then discarded. Inference requires only the saved
bridge and template-decoder directories, not the upstream repository or files.

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.prepare --config configs/bridge.yaml
python -m prosodia_lyricist.train --config configs/bridge.yaml --smoke-test
python -m prosodia_lyricist.train --config configs/bridge.yaml
```

Existing prepared IPA data can be reused; the data configuration and hashes stay
unchanged. Song grouping and the new source features are constructed when loading
the bridge dataset. Any exceeded source/target limit skips the entire song, with
per-reason counts; inference rejects overlong songs without truncation.

Preparation retains note arrays alongside the existing IPA annotations. It
preserves contrastive pretraining assignments for shared song IDs and duplicate
artist/title or audio identities. Conflicting assignments fail. Real training
requires both the pretrained checkpoint and the unchanged preparation manifest
hash. The upstream format records a manifest path rather than its historical
hash, so the supplied CSV must be the original one. These checks cannot infer
undocumented upstream exposure. The reused lyric checkpoint may have seen songs
from a different split; audit that overlap before claiming held-out performance
for the combined lyric system.

The smoke test runs at most two training and two validation batches with a tiny
template decoder. It imports the supplied melody checkpoint if configured;
otherwise it uses a tiny random melody tower and records that fact. Real
training rejects a missing contrastive checkpoint. Neither path downloads BART
weights or a tokenizer: the template bridge has its own fixed label vocabulary.

Training supports single-device FP32 or CUDA BF16, token-weighted gradient
accumulation including incomplete final groups, clipping, and early stopping
using validation cross-entropy. A frozen melody tower runs in evaluation mode
without building a gradient graph. Test data is never loaded for training or
checkpoint selection. Optimizer resume and distributed training are not included.

Each new run contains:

- `run.json`: configuration, split audit, checkpoint hash, dataset hashes,
  song/example/encoder-window counts, skipped examples by limit, and smoke-test status.
- `metrics.jsonl`: training/validation loss, teacher-forced token accuracy,
  learning rates, freeze state, and free-running validation metrics using IPA
  counts and predicted label history (exact phrase template accuracy, head
  accuracies, and note/IPA count diagnostics). Epoch logs also show free-running
  strength/length/pair accuracies; checkpoint selection still uses validation loss.
- `best/bridge.json` and `best/bridge_weights.pt`: label vocabulary, IPA rules,
  architecture, feature scheme, length limits, provenance, melody trunk and bridge weights.
  `best/run.json` retains the training context.

No original template checkpoint is loaded or updated during bridge training.

## Join the checkpoints at inference

```bash
python -m prosodia_lyricist.infer \
  --checkpoint checkpoints/dali/TEMPLATE_RUN/best \
  --bridge-checkpoint checkpoints/bridge/BRIDGE_RUN/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --output outputs/imagine-bridge.txt \
  --explanations outputs/imagine-bridge.explanations.json \
  --report-prefix outputs/imagine-bridge
```

`--checkpoint` remains the trained four-stream **template-decoder** directory.
The bridge is supplied separately via `--bridge-checkpoint`. Old template-only
inference continues to work when that argument is omitted. Python callers can
pass `bridge_checkpoint=...` to `prosodia_lyricist.infer.infer`.

MIDI input must be a monophonic melody with phrase-end markers. Note onsets on
a marker belong to that phrase, matching the existing convention. Unmarked tails
are retained. Tempo changes are converted to seconds before upstream feature
encoding. Learned inference does not use a beat-grid heuristic and accepts
non-4/4 meters. `--stress-source` is rejected with a bridge to avoid ambiguity.

Templates are decoded greedily against a fixed whole-song skeleton. Phrase note
counts determine its slots; each must fit `max_syllables`. Line IDs never reset
between encoder windows. Forced prefixes also advance the decoder cache so they
condition subsequent slots. Training still uses IPA counts, which may differ
from note counts because of melisma.
Malformed or overlong inputs fail instead of being truncated. Predicted templates
are passed through the existing `encode_source` function with the lyric checkpoint's
tokenizer. The lyric model's source/target limits and IPA correction remain active.
The bridge does not guarantee that generated lyrics exactly follow its template.

Inspect learned labels without loading the lyric decoder:

```bash
python -m prosodia_lyricist.infer \
  --bridge-checkpoint checkpoints/bridge/BRIDGE_RUN/best \
  --midi examples/imagine.mid --title Imagine --template-only \
  --report-prefix outputs/imagine-learned-template
```

JSON reports and explanations retain the original note arrays, predicted
syllable templates, note/syllable counts, bridge scope and checkpoint metadata.
They do not assign predicted syllables to individual notes. Report prosody-BLEU
measures lyric agreement with the **predicted** template, not bridge accuracy
against ground truth. Optional `--reference` lyrics supply scoring targets only
and do not change either model's inputs. Validation metrics from supervised
bridge data are the relevant first check of template prediction quality.

## Verification

Offline tests exercise song coordinates, exact local feature preservation,
global context, cached/full-decoder equivalence, causal shifting, padding,
gradients, trunk freezing, token-weighted accumulation, fixed skeletons, strict
contrastive import, source/target separation, split reuse, melisma retention,
window tails, preparation, training, reload, and combined inference with actual
template-model generation and perplexity scoring. They also cover tempo changes,
non-4/4 MIDI, reference isolation, and both report CLI paths. Tiny-model tests
verify the implementation; trained prediction quality still requires a real run.
