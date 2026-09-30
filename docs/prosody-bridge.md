# Contrastive melody → IPA template → lyrics

> Historical v1 design. The current v2 bridge uses supplied line prefixes and
> optional syllable counts, with no LINE_END prediction. See the README for usage.

The bridge replaces the heuristic MIDI-to-template step. It reuses the trained
melody trunk and 177-dimensional note encoding from `prosodia-direct`, learns
to predict the template decoder's source labels, and leaves the existing
four-stream lyric checkpoint intact. No HuBERT/audio model is loaded.

## Supervision and model

Each training example is a window of consecutive DALI phrases. Melody inputs
contain pitch, onset, duration, and phrase membership. Targets are the ordered
IPA stress/length pairs in each phrase. Words, lyric text, IPA labels and their
counts never enter the melody source. The title is supplied only to the existing
lyric model at inference.

The melody Transformer returns contextual **note vectors before pooled
contrastive projection**. A LayerNorm/linear/GELU adapter maps them to the small
template decoder's hidden size. Phrase embeddings are added after the melody
tower so the pretrained feature representation stays unchanged. A causal
Transformer decoder cross-attends to these note vectors and predicts an
eight-symbol vocabulary:

```text
PAD, BOS, EOS, LINE_END,
strong_long, strong_short, weak_long, weak_short
```

For example, two phrases with three and one syllables become:

```text
weak_short strong_long strong_short LINE_END weak_long LINE_END EOS
```

Training uses next-token cross-entropy with all targets shifted right behind
BOS. Padding is excluded from the loss. This jointly learns stress, vowel length,
and the decision to end a phrase. These are **source syllable labels**, distinct
from the existing lyric decoder's word-level explanation heads.

All notes, including melisma, are retained. There is no forced note-to-syllable
alignment. Training windows never cross songs or data splits. A final shorter
window is retained. Template prediction has local window context; the lyric
decoder still receives all predicted phrases as one complete song.

## Configure and train

Use the `prosodia-lyricist` Conda environment and `configs/bridge.yaml`.
Set these paths before preparing:

| Setting | Meaning |
| --- | --- |
| `model.melody_checkpoint` | Original contrastive checkpoint with `mlm_note_177d_v1` and `audio_pooling: note`. This is not a fine-tuned direct-lyrics checkpoint. |
| `data.pretraining_manifest` | Immutable CSV actually used for that contrastive run. |
| `data.lines_per_window` | Match the pretraining manifest's line count; default 2. |
| `data.prepared_dir` | Separate bridge dataset, default `data/dali-bridge`. |
| `model.max_notes` | Maximum notes per window; overlong training windows are counted and skipped. |
| `data.max_syllables` | Maximum IPA syllables per phrase; must fit the lyric checkpoint at inference. |
| `training.melody_unfreeze_epoch` | Zero-based epoch to fine-tune the trunk; 1 warms up the new decoder for one epoch, 0 starts immediately, null freezes throughout. |
| `training.learning_rate` | Adapter and template decoder learning rate. |
| `training.melody_learning_rate` | Separate, lower rate for the pretrained trunk. |

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
  window/song counts, skipped windows, and smoke-test status.
- `metrics.jsonl`: training/validation loss, teacher-forced token accuracy,
  learning rates, freeze state, and free-running validation metrics (exact
  phrase template accuracy, syllable-count accuracy/MAE, forced boundaries).
- `best/bridge.json` and `best/bridge_weights.pt`: label vocabulary, IPA rules,
  architecture, length limits, provenance, melody trunk, adapter and decoder.
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

Templates are decoded greedily. The supplied phrase count constrains generation
to that many nonempty phrases. A learned `LINE_END` determines each phrase's
syllable count; counts are bounded by `max_syllables`. If a boundary must be forced
at that cap, its zero-based phrase index is reported in `bridge.forced_line_endings`.
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
syllable templates, note/syllable counts, bridge checkpoint and forced boundaries.
They do not assign predicted syllables to individual notes. Report prosody-BLEU
measures lyric agreement with the **predicted** template, not bridge accuracy
against ground truth. Optional `--reference` lyrics supply scoring targets only
and do not change either model's inputs. Validation metrics from supervised
bridge data are the relevant first check of template prediction quality.

## Verification

Offline tests exercise causal shifting, padding, gradients, trunk freezing,
token-weighted accumulation, variable syllable counts, forced boundaries, strict
contrastive import, source/target separation, split reuse, melisma retention,
window tails, preparation, training, reload, and combined inference with actual
template-model generation and perplexity scoring. They also cover tempo changes,
non-4/4 MIDI, reference isolation, and both report CLI paths. Tiny-model tests
verify the implementation; trained prediction quality still requires a real run.
