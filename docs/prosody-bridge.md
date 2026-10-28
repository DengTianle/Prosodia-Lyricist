# Global melody → IPA template → XAI lyrics

The bridge uses one bidirectional melody Transformer over every note in a song,
followed by shared stress and vowel-length heads. BART/XAI receives the resulting
discrete template and is unchanged. Bridge checkpoints are v10; older bridges
and masked-pretraining v1 checkpoints require retraining. Prepared note arrays
remain reusable, including files with obsolete window-size metadata.

## Source and supervision

The original 177-D representation is preserved: 128 pitch-displacement bins and
a sign bit, 24 log-duration bins, and 24 log-IOI bins. The pitch reference and
min/max normalization now cover the entire song. Global sinusoidal positions and
phrase embeddings precede attention. No local windows, reassembly, separate
39-D feature stream, second encoder stack, or query decoder remain.

Two heads predict `strong/substrong/weak` and `long/short` by default. Observed
DALI word/note links gather first-note predictions into IPA targets. With
`alignment_policy: strict`, mismatched words are masked; `prefix` pairs their
first `min(sung, IPA)` slots. All notes remain encoder context. Gold labels,
counts, alignment and lyrics never enter the encoder. MIDI inference emits one
pair per note, using phrase markers and retaining an unmarked tail.

Training uses `CE_stress + CE_length`. Optional `length_class_weights: [long,
short]` weights length CE and checkpoint selection; ordinary CE is still logged.
Gradient accumulation weights supervised slots across the entire group. Dataset
limits skip whole songs; inference rejects overlong input. Full attention uses
quadratic memory, so batch size should reflect song lengths.

## DALI training and initialization

Use `configs/bridge.yaml`. `model.melody_encoder` controls the single encoder's
width, depth, heads, FFN, dropout and positional capacity. `max_song_notes` must
fit that capacity. `melody_initialization` supports:

- `random`: train the configured encoder from scratch (default).
- `masked`: import a global masked-pretraining `best/` directory via
  `melody_checkpoint`, with fresh stress/length heads. Architecture and source
  limits must match, as must prepared train/valid checksums.
- `bridge`: fine-tune a v10 `best/` directory via `melody_checkpoint`, retaining
  its heads and architecture. Conflicting architecture/limit overrides fail.

`training.melody_unfreeze_epoch: 0` trains the main encoder immediately; null
freezes its weights. `melody_learning_rate` controls that encoder separately
from heads and conditioning adapters (`learning_rate`). A frozen encoder still
propagates gradients to its trainable conditioning inputs.

## Optional meter fine-tuning

DALI has no observed bar grid, so meter defaults to absent. A prepared record may
add `melody.meter`, with one object or null per note:

```json
[{"numerator": 6, "denominator": 8, "beat_position": 1.5}, null]
```

`beat_position` is zero-based within the bar, in denominator-note units; for
6/8 it lies in `[0, 6)`. The encoder receives sine/cosine bar and beat phases,
log2 numerator/denominator, and a separate observation mask. Missing meter,
including mixed annotated/unannotated songs, contributes exactly zero. No beats
are inferred from DALI durations or IPA stress. Labels and word/note alignment
remain required for supervised fine-tuning; update the prepared manifest's
checksums when adding annotations.

To fine-tune on such prepared data, use the normal training command with:

```yaml
model:
  conditioning: bridge
  melody_initialization: bridge
  melody_checkpoint: ../checkpoints/bridge-xai/RUN/best
  meter_conditioning: true
```

Keep the normal data/training sections, pointing data settings at the annotated
corpus's manifest. The new meter adapter's final projection starts at zero,
preserving the base model initially. MIDI inference extracts explicit time
signatures and tick-based bar positions, with each signature change starting a
new bar grid. Notes before the first signature remain unannotated; pickups
requiring another grid origin should be explicitly annotated in prepared data.

## Optional contrastive fusion

The main encoder does not depend on a contrastive checkpoint. To add one during
initial training or when fine-tuning a v10 bridge:

```yaml
model:
  contrastive_fusion:
    checkpoint: ../../try-contrastive/checkpoints/large_2line_16s/best.pt
    unfreeze_epoch: null
    learning_rate: 0.000002
```

The imported melody trunk processes the **same whole-song 177-D sequence**.
Its note vectors and pooled song vector are concatenated, projected to the main
encoder width, and added before global attention through a learned `tanh` gate.
The gate starts at zero. The audio tower and contrastive projection are unused.
The trunk starts frozen, including dropout; the gate and adapter train normally.
Set `unfreeze_epoch: 0` or a later zero-based epoch to tune the trunk separately.
Global normalization/longer context differ from the original two-line training
distribution; the benefit of fusion needs an ablation against the plain encoder.

Set `data.pretraining_manifest` to the matching upstream CSV before preparation.
Training retains song-identity leakage checks: strict mode requires the unchanged
audited CSV; warning mode records missing/conflicting audits. Window sizes no
longer need to match. Saved bridges include fusion weights, architecture and
provenance, so inference needs no upstream files. When fine-tuning an already
fused bridge, omit `contrastive_fusion.checkpoint` to retain its saved tower.

## Masked pretraining and evaluation

`configs/bridge_pretrain.yaml` trains the same global encoder from scratch,
reconstructing pitch displacement, sign, duration and IOI at masked notes. Only
the song's first note is protected as a reference. Selected features and
successor IOIs are hidden. Neither meter, lyrics nor IPA labels enter this
objective. Validation masks are fixed, and validation CE selects `best/`.

Evaluation schema v6 reports aligned slot CE/accuracy and classification
statistics. Exact-template accuracy and BLEU use fully supervised phrases;
coverage and masked targets are reported. Duration controls use corpus, song
and phrase means on the same note anchors. The README retains the training,
evaluation and MIDI-to-BART commands.
