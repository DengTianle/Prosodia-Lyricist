# Direct melody conditioning with the template decoder

This implements option A on `template-decoder`. It retains four aligned outputs
(lyrics, syllable count, stress, length), compound decoder feedback, four losses,
and IPA correction before the next word. The source changes from a lyric-derived
template to actual melody notes.

```text
Melody phrase windows → pretrained melody Transformer → one vector per note
    → learned LayerNorm/MLP adapter → ordered song/paragraph source
    → BART encoder → four-stream decoder → lyrics and prosody explanations
```

## Contextual note embeddings and pooling

The upstream encoder first converts each note to 177 features describing pitch
displacement, duration and inter-onset interval. A Transformer processes the
window. Its output contains one vector per note, incorporating information from
other notes in that window. These are the contextual note embeddings, taken
after the encoder's output LayerNorm.

The Transformer also includes a CLS token. With `pooling="cls"`, its output
summarizes the window for the contrastive loss. With `pooling="mean"`, the summary
is the masked average of the note output vectors. Both modes still produce all
individual note vectors. The contrastive projection is trained on the summary.

This integration uses `encode(..., project=False).note_embeddings`; it does not
switch to mean pooling or project each note through the pooled projection head.
Switching only pooling on identical weights does not change the note vectors.
Training another upstream checkpoint with mean pooling can change the weights
and therefore those vectors. That is a valid experiment. Upstream should record
its actual choice as `args.melody_pooling`; missing metadata uses the existing
CLS default. No changes were made to the adjacent repository.

The vendored encoder comes from the integration pinned to `try-contrastive`
revision `ff6247f2613fd5e1d1f06c44b73f9dadfa6cb257` (two-pool). Import validates
the `mlm_note_177d_v1` representation tag, `audio_pooling: note`, architecture
metadata and all melody weights. HuBERT is never instantiated. The upstream
checkpoint is memory-mapped on CPU; only the melody trunk is retained. Legacy
frame checkpoints are rejected.

## Song context and source inputs

`data.unit: song` produces one decoder target per complete song. `paragraph`
uses contiguous DALI paragraph IDs, with split assignments still made by song.
There is no automatic conversion to paragraph training when a song is too long.

The melody encoder processes non-overlapping windows of `data.lines_per_window`
phrases. Match the pretraining manifest's line count; real training checks it
when available. Each window retains upstream feature normalization and local
positions. A final shorter window is retained. BART receives all adapted notes
in order, with BOS, an optional title, sentence markers and EOS. BART adds its
own positions and attends across the complete song/paragraph. Melody window
boundaries do not reset lyric decoder context.

Source construction reads notes, phrase boundaries and the optional title only.
IPA labels, lyric words and reference syllable counts are targets, never source
features. Melisma notes are retained individually: one note need not correspond
to one syllable, word or BPE piece.

DALI supplies phrase boundaries during training. MIDI inference expects phrase-end
markers, using the existing convention that a note starting exactly on a marker
belongs to that phrase. Tempo changes are respected. Notes after the final marker
form another phrase. For paragraph models, supply one paragraph as the MIDI input;
MIDI paragraph segmentation is not inferred.

## Configure and run

Use the `prosodia-lyricist` environment. On this machine its interpreter is
`/opt/anaconda3/envs/prosodia-lyricist/bin/python`.

Edit `configs/melody.yaml`:

- Set `data.pretraining_manifest` to the immutable CSV actually used to train
  the melody encoder, **before preparation**. Shared song IDs and duplicate
  artist/title or audio identities retain their pretraining splits. Other groups
  use the seeded split rule; conflicting assignments fail explicitly.
- Set `model.melody_checkpoint` to the matching current two-pool checkpoint.
- Set `data.lines_per_window` to its pretraining window size.
- Optionally set `model.template_checkpoint` to a trained four-stream
  `template-decoder` `best/` directory. All BART weights, compound embeddings,
  projections and auxiliary heads are copied. Lyrics-only checkpoints are rejected.
- With a trained template checkpoint, `freeze_decoder_epochs: 1` enables adapter
  warmup. Without one, the default `0` immediately trains the new compound features
  and pretrained BART. `melody_unfreeze_epoch: null` keeps the melody tower frozen;
  an integer enables later fine-tuning at its separate learning rate.

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.prepare --config configs/melody.yaml
python -m prosodia_lyricist.train --config configs/melody.yaml --smoke-test
python -m prosodia_lyricist.train --config configs/melody.yaml
```

Preparation uses a separate `data/dali-melody` directory, retaining note arrays
alongside IPA targets. Template-only preparation is unchanged. Invalid pitch or
timing and unusable lyric lines reject the whole song. Paragraph mode requires
valid paragraph IDs rather than guessing stanza boundaries.

Smoke tests use tiny random models only where checkpoints were not supplied,
FP32 and at most two batches per split. Random initialization is recorded. Real
training requires a melody checkpoint and unchanged pretraining-manifest hash.
Upstream checkpoints store a manifest path, not its historical hash: supplying
the actual original manifest remains the user's responsibility. Undocumented
pretraining exposure and text-corpus overlap in a template checkpoint cannot be
automatically ruled out.

## Memory and training behavior

The H100 configuration uses BF16, microbatch 2, four-step gradient accumulation,
length-bucketed batches and non-reentrant BART activation checkpointing. Losses
are accumulated by valid target-event counts, including the last partial update
group. Use `precision: fp32` on CPU/MPS; BF16 here requires supported CUDA hardware.

Melody, adapter and decoder use separate learning rates. Decoder freezing covers
the complete template model, including the BART encoder and compound heads.
Frozen modules use evaluation mode. Frozen BART still passes gradients to the
adapter; the frozen melody encoder runs without an autograd graph.

BART's source and target limits default to 1,024 positions independently. The
target contains BPE pieces plus a `<word_end>` event for every pronounceable word.
Complete units exceeding source, target, melody-window or sentence-marker limits
are skipped and counted in `run.json`. Nothing is truncated. Check these counts
when choosing song versus paragraph training.

`metrics.jsonl` records four losses, optimizer steps, learning rates, freeze state
and CUDA peak allocated/reserved GiB for each epoch phase. Tiny smoke-run memory
is not a full-model capacity benchmark. Local tests do not measure H100 memory
or throughput. Training is single-device; DDP, optimizer resume and embedding
caches are not implemented.

## Generation

```bash
python -m prosodia_lyricist.infer \
  --checkpoint checkpoints/melody/RUN_ID/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --output outputs/imagine-direct.txt \
  --explanations outputs/imagine-direct.json
```

Saved checkpoints include the melody trunk, adapter, four-stream model, tokenizer,
window settings and provenance. Inference does not need the adjacent repository
or original melody checkpoint. The JSON contains source notes, predicted/corrected
word prosody, generated streams, completion status and source/window counts.
`--no-prosody-correction` remains an ablation. Template stress heuristics and
`--report-prefix` are unsupported for direct checkpoints; use `--explanations`.

Conditioning does not enforce exact syllable counts, phrase counts or note/word
alignment. Trained quality still requires evaluation, including shuffled-melody
and random-encoder comparisons.

## Verification

Offline regressions cover source/target separation, window normalization, song
packing, paragraphs and length boundaries, melisma, split reuse, CLS/mean imports,
strict template transfer, freezing/unfreezing, all four losses, causal shifts,
padding, accumulation equivalence, reload and actual cached IPA correction.
End-to-end tests prepare DALI, train with both random smoke initialization and
imported template/melody weights, then generate from MIDI with tempo changes and
an unmarked trailing phrase.
