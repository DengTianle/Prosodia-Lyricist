# Prosodia Lyricist

Prosody-conditioned lyric generation with BART, trained on the **DALI music
annotation dataset**. This branch trains a global melody encoder with direct note
classification into IPA templates, then feeds its discrete predictions to the original
XAI-Lyricist `bestM2LCkpt.pt`. Each note attends to the complete song and predicts stress and vowel
length with separate CE losses. The bridge has no label feedback or autoregressive decoding. `train.sh`,
`prep.sh`, and `configs/bridge.yaml` select this workflow; see the
[bridge commands](#learned-melody-to-template-bridge) below. XAI's single-stream
lyric decoder and compound meter/length/remainder source embeddings are preserved.

## Setup

Use the existing Conda environment (its name on this machine is
`prosodia-lyricist`):

```bash
conda activate prosodia-lyricist
python -m pip install -e '.[midi,dev]'
# Default IPA mode uses modern prosodic (tested with 3.10.0).
python -m pip install -e '.[ipa]'
# Optional: keep downloaded Hugging Face files inside this project.
export HF_HOME="$PWD/data/huggingface"
```

Python 3.10+ is required. `midi` adds MIDI inference; `dev` adds pytest and Ruff.
Preparation needs no audio downloads, MIDI files, DALI Python package, or custom
Transformers fork. Default IPA preparation uses `prosodic>=3.10,<4`, including
its current word-token/wordform API and explicit syllable IPA strings. Install
its tokenizer resources once if missing:

```bash
python -m nltk.downloader punkt punkt_tab
```

The paper adapter selects the first modern pronunciation for each word, preserves repeated
words and punctuation, and does not run metrical or syntactic parsing. The
lexical/unknown alternatives do not need this dependency. The first training
run fetches the BART tokenizer and pretrained weights from Hugging Face.

## Prepare DALI

Place the official horizontal `.gz` annotation files in `DALI_v2/`, or set
`data.dali_dir` in [`configs/bridge.yaml`](configs/bridge.yaml). DALI JSON exports
with `info` and `annotations` fields are also supported. Paths in the YAML are
relative to the configuration file, so commands also work outside the repo.

```bash
prosodia-prepare --config configs/bridge.yaml
```

This writes `data/dali-xai/{train,valid,test}.jsonl`, `manifest.json`, and
`rejected.jsonl`. Each record is one complete song with an ordered `lines` list;
the manifest records configuration, pronunciation version, song/group split
assignments, song and line counts, and file checksums. Schema version 5 records
repo/paper IPA rules and pronunciation provenance. Repo-mode XAI training requires
new preparation; schema-4 paper data remains readable by earlier models.
Preparation can be rerun and replaces only these generated outputs. Raw data
and old local experiment outputs are ignored by Git.

The default configuration selects English songs and uses a seeded hash for
approximately 80%/10%/10% train/validation/test song groups. When a contrastive manifest is configured, groups exposed to its training
split are assigned to downstream train. All
lines from a song stay together. Entries sharing normalized artist/title,
the same audio ID, exact encoded melody inputs, or normalized full lyrics
(at least 100 alphanumeric characters) are grouped transitively. Optional meter
and target labels cannot hide an identical melody. The metadata-only
`--check-splits` preflight runs before IPA; final content grouping follows extraction.
This does not identify every cover or near-duplicate recording in DALI.
`min_ncc` optionally filters DALI's alignment score; the default 0.0 adds no
quality threshold. Unknown-language songs are excluded by the English filter.

Existing exports need an explicit content audit; their file checksums and song IDs
do not prove split independence. Audit DALI and Lakh together and optionally create
repaired copies in a **new** directory:

```bash
python -m prosodia_lyricist.split_integrity \
  --prepared-dir data/dali-xai --prepared-dir data/lakh-bridge \
  --output outputs/content-split-audit.json --repair-root data/content-clean
```

Repair moves connected groups to train before validation before test, preserving
the source exports. Point training configs at the repaired copies and start from
random weights; repartitioning cannot remove an old checkpoint's training exposure.
Training and masked pretraining reject content conflicts. New supervised bridges
record ancestor training and validation identities; fine-tuning rejects ancestor
training in a held-out split and ancestor validation reused as a test set. Legacy
bridges without that provenance remain readable for inference, but cannot serve
as verified fine-tuning parents. Evaluation reports split integrity separately
from prediction metrics, including unverified legacy ancestry. Near duplicates
and shared compositions still require independent review.

To select an exact subset, set `data.song_ids_file: ../song_ids.txt` in the YAML
(one exact annotation `info.id` per line; blank lines and duplicates are ignored).
This path is resolved relative to the YAML, and selection uses annotation IDs,
not filenames. The manifest stores the sorted requested IDs, the list-file SHA256,
missing IDs, and IDs excluded by language/quality/annotation checks. The list is
an allowlist, not an override of those checks or a specification of split membership.

Equivalent preparation overrides are available:

```bash
prosodia-prepare --config configs/bridge.yaml --english-only --song-ids-file song_ids.txt
```

CLI paths are relative to the current directory. Before training, save matching
values in the YAML: training checks its data settings against the manifest.
`language: all` disables language filtering; IPA pronunciation remains English.
`--limit` limits scanned files, so the resulting missing-ID list may be partial.

### What the model learns from DALI

- **Line and word membership:** DALI's parent indices link notes to words and
  words to lines. The target is the joined parent-word text.
- **Default IPA labels:** `stress_source: ipa` and `prosody_mode: actual` use
  historical Prosodic 1.5 dictionary-first pronunciation, with modern Prosodic
  tokenization and explicit fallback. Primary/secondary/unstressed syllables map
  to `strong/substrong/weak`; only the IPA `ː` mark means long. Counts use IPA
  syllables even when note counts differ. Unpronounceable words reject the song.
  Punctuation tokens are retained with zero syllables. Paper-mode binary labels
  remain available for template comparisons, with separate prepared data.
- **Optional sung features:** `stress_source: lexical` retains the previous
  DALI sung-count / CMUdict stress / relative-duration length mode.
  `stress_source: unknown` uses sung counts and duration with unknown stress.
  Empty or `~` continuation notes merge into the preceding syllable. Durations
  sum sounding intervals without gaps; above-line-mean durations are long.
  Empty parent words containing only explicit `~` markers extend the preceding
  word within the same line, after validating their original timing bounds.
  Missing lyric text with ordinary notes and continuations starting a line still
  reject the song; neighboring text fragments are never guessed or joined.
- **Alignment provenance:** `sung_syllables` retains DALI note timing separately
  from the IPA template; `words` retains deterministic per-word syllable counts.

**DALI has no bar-line annotations.** No tempo, meter, downbeats, or bar lines
are inferred from its annotation frame rate. Pronunciation stress is a lexical
proxy, not measured musical stress. Default training uses IPA vowel length,
not DALI durations, matching the original text-derived template approach. The bridge receives pitch, onsets, durations and phrase membership; XAI receives
its predicted template. Optional sung preparation remains dependent on DALI
syllable and duration quality and does not provide the current bridge's IPA labels.

Invalid indices, orphan continuations, overlapping/out-of-bounds intervals,
empty words, and lines above `max_syllables` (default 40 per line) are reported.
Any unusable line rejects the whole song, preserving complete context without
joining across missing lines. Bridge dataset limits skip and count whole songs,
without truncation. The separate XAI source and target limits are enforced at
inference. Test data is held out from training and checkpoint selection.

For a preparation smoke test, `--limit 100` scans the first 100 files. Use a
separate `prepared_dir` if you want to keep the complete prepared dataset;
small subsets may have empty splits, in which case training fails clearly.

## Learned melody-to-template bridge

The bridge encodes a whole song in one bidirectional Transformer pass. It uses
the same 177-dimensional pitch-displacement, log-duration and inter-onset-interval
representation, normalized once across the song. Phrase embeddings enter before
attention; shared stress and vowel-length classifiers read each contextual note.
There is no window splitting/reassembly or second song encoder.

```text
whole-song 177-D notes + phrase embeddings + optional meter / contrastive context
  → global melody Transformer → note stress/length heads → discrete IPA template
  → original XAI-Lyricist template encoder and lyric decoder → lyrics
```

DALI training starts from scratch by default, or from global masked-note
pretraining. Optional meter conditioning and a gated contrastive tower can be
added when fine-tuning a saved global bridge. See the
[configuration recipes](docs/prosody-bridge.md) for both.

Observed DALI word/note links map the first note of each sung-syllable group to
an IPA slot. Gold labels and this mapping select losses and scores; they never
enter the encoder or classification heads. Both supervised losses backpropagate
through the global melody encoder when it is unfrozen:

```text
loss = CE_stress + CE_length
```

Set `training.length_class_weights: [2.0, 1.0]` to weight long/short length CE;
the default `null` is unweighted. `loss`/`components` continue to report ordinary
CE. Weighted runs also record `objective_loss`/`objective_components`, and select
`best/` by the weighted validation objective. Masks and stress CE are unchanged.

The default `data.prosody_mode: actual` preserves XAI's three stress classes
(`strong`, `substrong`, `weak`) and binary vowel length, where only the IPA
long mark means long. Prefixes and padding contribute no loss. The title and
lyric text never enter the bridge; the title is supplied to XAI at inference.
The XAI checkpoint is neither loaded nor updated during bridge training.

Training and evaluation gather note predictions into the reference IPA skeleton.
`model.alignment_policy: strict` masks whole words with unequal sung/IPA counts;
`prefix` instead pairs their first `min(sung, IPA)` positions and ignores the
unmatched tail. Alignment always restarts at word boundaries. Unresolved text
alignment is masked. All notes remain encoder context, including continuations
and masked words. MIDI inference emits one pair per note and retains an unmarked
tail as a final phrase. There is no role or count predictor.

Configure `configs/bridge.yaml` for the encoder size and training device. For
contrastive fusion, set `model.contrastive_fusion.checkpoint` and
`data.pretraining_manifest` before preparation so upstream training identities
remain in downstream training. Duplicate song/audio groups also stay together.
Limits are 2,048 notes, 256 lines, and 4,096 IPA/structural positions per song.
Overlong training songs are skipped whole and counted; inference rejects them.
Global attention memory grows quadratically with note count; adjust batch size
for the available device.

```bash
conda activate prosodia-lyricist
# Install historical pronunciation assets once if absent.
python -m prosodia_lyricist.historical_ipa
python -m prosodia_lyricist.prepare --config configs/bridge.yaml --check-splits
python -m prosodia_lyricist.prepare --config configs/bridge.yaml
python -m prosodia_lyricist.train --config configs/bridge.yaml --smoke-test
python -m prosodia_lyricist.train --config configs/bridge.yaml

python -m prosodia_lyricist.bridge_eval \
  --checkpoint checkpoints/bridge-xai/BRIDGE_RUN/best \
  --prepared-dir data/dali-xai --split valid --device cuda --precision bf16 \
  --output outputs/bridge-eval.json

python -m prosodia_lyricist.infer \
  --checkpoint checkpoints/bestM2LCkpt.pt \
  --bridge-checkpoint checkpoints/bridge-xai/BRIDGE_RUN/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --report-prefix outputs/imagine-bridge
```

An imported XAI `xai.json` directory can replace the original checkpoint file.
Use `--template-only` with the bridge checkpoint to inspect predictions without
loading XAI. Reference lyrics are used only for scoring. XAI's tokenizer/config
overrides and decoding controls remain available.

Existing `data/dali-xai` data can be reused when its preparation configuration
matches. New schema-v6 preparation stores word/note links. Older files recover
them in memory from `data.dali_dir`, without rerunning IPA or changing the files;
raw annotation checksums are recorded in the run. Evaluation accepts `--dali-dir`
to override a cluster path. Old binary paper-mode labels cannot supervise an XAI
bridge. For upstream CSV changes, `training.pretraining_audit_mode: warning`
retains prepared-file integrity checks and records missing/conflicting audit
information; `strict` requires the unchanged original CSV.

Evaluation schema v6 reports one `prediction` block: slot-weighted loss, separate
loss components, head/pair accuracies, exact phrase accuracy, prosody-BLEU, and
note/IPA count diagnostics. Slot metrics use only aligned targets. Exact-template
accuracy and BLEU use fully supervised phrases, with coverage reported explicitly.
Unsmoothed BLEU-4 scores phrases shorter than four slots as zero. Duration
baselines use the same first-note anchors and target masks. Checkpoint selection
uses validation CE over supervised slots.
The `classification` block adds per-head reference/prediction counts, confusion
matrices, mean probabilities, per-class recall, balanced accuracy, and macro F1.
`label_entropy` is the empirical reference entropy in nats; it helps identify
constant-output behavior. Epoch logs also show predicted-long fraction and long
recall, so majority-class collapse is visible without opening the JSON.
Validation additionally reports long-vs-short ROC AUC and average precision:
these reveal useful ranking even when every argmax prediction is still short.

New checkpoints use bridge format v10; masked pretraining uses format v2.
Older window/query bridge and masked checkpoints require retraining. Prepared
note arrays remain reusable: obsolete `lines_per_window` metadata is ignored.
The BART/XAI architecture and checkpoint format are unchanged.

### Metered training on Melistas/Lakh

The calibrated alignment export in `outputs/lakh-alignment-audit` can be prepared
and trained with the same bridge commands:

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.prepare --config configs/bridge_lakh.yaml
python -m prosodia_lyricist.train --config configs/bridge_lakh.yaml
# Slurm wrappers also accept the configuration path:
sbatch prep.sh configs/bridge_lakh.yaml
sbatch train.sh configs/bridge_lakh.yaml
```

`data.lakh_file` points to `deduplicated-orthographic.jsonl.gz`; outputs go to
`data/lakh-bridge`. The preparer retains the authors' processed notes and handles
melisma with the same word-local continuation rules as DALI. Reconstructed words
use the XAI `actual` historical IPA policy, including secondary stress and
long-mark-only vowel length. Strict word-count mismatches mask supervision while
retaining melody context. Whole songs with unusable lines or over-limit sequences
are rejected with reasons.

The default Lakh configuration requires explicit meter at every melody note and
enables `model.meter_conditioning`. Bar phase uses MIDI tick zero/signature changes
as grid origins, matching MIDI inference; pickups are not independently corrected.
Unresolved tempo or conflicting meter metadata rejects the song. The manifest
records source hashes, original note indices/ticks, pronunciation, alignment
coverage, and exclusions. Seeded splits group identical source hashes and exact
full lyrics of at least 100 normalized characters; no DALI overlap filter is applied.
The phonetic-only export does not supply the required spelling alignment.

For local CPU/MPS runs, set `training.device` to `auto` and `precision` to `fp32`.
To fine-tune an existing bridge, set `melody_initialization: bridge` and
`melody_checkpoint` to its `best/` directory, preserving its architecture and limits.

### Self-supervised melody pretraining

`configs/bridge_pretrain.yaml` reconstructs masked musical attributes: pitch
displacement, pitch sign, duration bin, and inter-onset interval bin. It reads only
melodies from the existing prepared train/valid splits; no IPA labels, alignment,
raw DALI files, or lyric checkpoint are needed. By default it starts from scratch
and trains the global encoder and phrase embeddings together at `1e-4`.

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.bridge_pretrain --config configs/bridge_pretrain.yaml --smoke-test
python -m prosodia_lyricist.bridge_pretrain --config configs/bridge_pretrain.yaml
```

Then set `model.melody_initialization: masked` in `configs/bridge.yaml` and point
`model.melody_checkpoint` to `../checkpoints/bridge-pretrain/RUN/best`. Keep its
encoder architecture and source limits equal to the pretraining configuration.
Run the normal bridge training command: it imports the full encoder with fresh
stress/length heads. The resulting bridge uses the usual evaluation and MIDI
commands. Import requires the same prepared train/valid checksums to preserve the
held-out split.

Masking selects 15% of non-anchor notes (at least one per song), zeroing their
177-D inputs and hiding dependent successor IOIs. Only the song's first note
stays visible as its pitch reference. Loss is the
sum of four CEs over masked notes only. Validation masks stay fixed; logs report
each reconstruction loss/accuracy, class counts, and majority baselines.
Validation CE selects `best/`.

## XAI lyric inference and reports

Pass the original `checkpoints/bestM2LCkpt.pt` or an imported XAI directory as
`--checkpoint`, and the trained bridge's `best/` directory as `--bridge-checkpoint`.
The XAI encoder/decoder stays unchanged. To import it once:

```bash
python -m prosodia_lyricist.xai_model --checkpoint checkpoints/bestM2LCkpt.pt \
  --output-dir checkpoints/xai-original --local-files-only
```

The author's tokenizer and BART config are absent from the public assets. Without
`--tokenizer-dir` / `--bart-config`, import recovers observed special-token IDs,
extends the count-token mapping to 1..40, and uses BART-base settings not inferable
from weights. These assumptions are recorded in provenance; strict tensor loading
establishes architecture compatibility, not exact numerical reproduction.

Choose a monophonic melody using `--track` (default 0). Markers specify phrase
ends. The current bridge retains an unmarked trailing phrase and requests one
slot per note. XAI receives all phrases and the title in one source sequence;
generated periods become lyric line breaks. It does not enforce the requested
line or syllable counts. Overlong inputs fail without truncation.

`--top-k 1` is greedy; larger values use top-k temperature sampling. Optional
`--no-repeat-ngram-size` and `--repetition-penalty` override XAI's defaults (0 and
1). No prosody correction or auxiliary decoder streams are used. The generation
budget defaults to 1023 tokens for the original checkpoint.

`--report-prefix PREFIX` writes new Markdown and JSON files with predicted
input templates, output IPA, note details, phrase counts, prosody-BLEU, and
conditional BPE perplexity. `--reference lyrics.txt` adds reference PPL and IPA
(one nonempty line per input phrase); references never enter generation.
Generated-text PPL is a self-score. Template-agreement BLEU is distinct from
reference accuracy. `--explanations details.json` saves completion and source
metadata. See [evaluation conventions](docs/evaluation.md).

Without a bridge, `--template-only` inspects heuristic MIDI templates. The default
repo (`actual`) heuristic uses marked phrases only; `--prosody-mode paper`
selects the explicit supplement comparison, retaining the tail. Heuristic
`--stress-source` overrides require paper mode.

### Rolling Stone MIDI pilot

Evaluate MIDI prosody against the reference lyrics in `data/bridge-eval`:

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.midi_eval --output outputs/bridge-eval-midi.json
# Add a learned bridge (loaded once for the entire evaluation):
python -m prosodia_lyricist.midi_eval \
  --bridge-checkpoint checkpoints/bridge-xai/RUN/best \
  --output outputs/bridge-eval-comparison.json
```

The default MIDI methods are `actual`, `supplement`, and `heuristic`; use
`--methods actual bridge` or `--methods bridge` to select methods. References use
the checkpoint's IPA rules, or `actual` without a bridge; `--prosody-mode paper`
selects binary stress and diphthong-aware length. Numeric lyric markers such as
`(1)` and `(2)` are removed before pronunciation. Corpus stress annotations are
not IPA references and do not supply vowel-length labels.

The JSON report contains per-song/per-phrase results and micro accuracy for
strength, length, and their joint correctness. Phrases and syllables are paired
by position, with missing/extra slots counted as errors; equal-count phrases are
also reported separately. All methods retain unmarked MIDI tails. Inference uses
one slot per note and receives no reference text or counts. Failures (including
unsupported non-4/4 supplement inputs) are listed with coverage; `common_subset`
compares methods on the same successfully evaluated songs. `--songs SONG ...`,
`--limit N`, `--track`, and `--device` control the run. These are lyric IPA
agreement scores; count differences can shift comparisons within a phrase.

### Full pipeline on the 25-song set

`pipeline_eval` compares MIDI algorithm inputs and bridge predictions through the
same unchanged BART lyric decoder. Both models load once for the full run:

```bash
python -m prosodia_lyricist.pipeline_eval \
  --checkpoint checkpoints/xai-original \
  --bridge-checkpoint checkpoints/bridge-xai/20261022T151203578546Z/best \
  --output outputs/bridge-eval-pipeline.json
```

Defaults are `--data-dir data/bridge-eval`, `--methods actual bridge`, repo IPA
rules, and greedy decoding (`--top-k 1`). Other MIDI algorithms can be selected
with `--methods actual supplement heuristic bridge`. The decoder uses the same
whole-song generation and limits as `infer`; no chunking or truncation. All input
methods retain unmarked tails for a matched comparison. `--songs`, `--limit`,
`--track`, `--device`, and the usual generation overrides are supported.

Reports separate template/reference agreement, generated-lyric/input-template
adherence, and generated-lyric/reference agreement, using prosody-BLEU and
strength/length/joint accuracy. They also include teacher-forced reference PPL,
generated-text self PPL, count mismatches, completion flags, and coverage. Reference
phrase/count mismatches are scored rather than rejected; references supply targets
only. No training-overlap filter is applied. Aggregate PPL sums NLL and token counts;
the summary compares the common successful songs, with separate common PPL coverage.

The command writes detailed `.json`, a short `.md` summary, and `.songs.jsonl`
flushed after each song. Use a new output path for each run. JSON includes templates,
lyrics, IPA, token IDs, file/checkpoint hashes, seeds, and settings for inspection.

By default, it also writes readable reports for **3 random songs** in
`OUTPUT.details/`, using the same sampled songs for every input method. Start at
`index.md`: each song has an `actual.md` / `bridge.md` report in the same format as
`infer` (lyrics, MIDI notes, per-phrase counts, prosody patterns, and word/IPA tables),
matching `.lyrics.txt` files, and `reference.lyrics.txt`. The reports also compare
input/reference, generated/input, and generated/reference metrics.

Use `--report-count K` to change the sample size (`0` disables reports), or
`--report-songs SONG ...` to choose songs from the evaluated set. Sampling uses the
evaluation seed; `--report-seed N` changes only the report sample. Samples are capped
at the number of songs with a successful generation. `--report-dir` selects a new
destination folder. `--songs` still controls which songs are actually evaluated.

Export reports from existing pipeline JSON **without rerunning models or IPA**:

```bash
python -m prosodia_lyricist.pipeline_eval \
  --from-report outputs/bridge-eval-pipeline.json \
  --report-songs a_day_in_the_life hotel_california \
  --report-dir outputs/pipeline-selected
```

Omit `--report-songs` to export the default random three. Report-only mode leaves
the saved JSON unchanged and does not require checkpoints or the original MIDI files.

Architecturally, `bridge_eval` scores held-out bridge labels, `midi_eval` compares
MIDI-to-template producers, `infer.TemplateDecoder` owns reusable BART inference,
and `pipeline_eval` composes those producers with the decoder and dataset scoring.
`pipeline_reports` adapts saved results to the shared inference report renderer.

With `melismas/`, `stress-format-space/`, and `rs_lyrics/` under
`data/rolling-stone`, convert two songs marked `absent` in the DALI overlap CSV:

```bash
conda activate prosodia-lyricist
python scripts/prepare_rolling_stone_midi.py --limit 2
# Choose songs explicitly and use a new output directory for another run:
python scripts/prepare_rolling_stone_midi.py --songs 1999 a_day_in_the_life \
  --output-dir outputs/rolling-stone-pilot
```

The default output is `data/rolling-stone/midi-pilot/`: melody MIDIs with 4/4
meter and phrase-end markers, one-line-per-phrase reference text, annotation JSON
with syllable-to-note links, and a manifest with source hashes and skipped songs.
Exactly one note per `.str` syllable is emitted: only the first note of each
melisma is retained. The full `.mel` validates pitch/onset and supplies exact
timing, rests and meter; omitted continuation counts are recorded in annotations.
Nonempty lyric lines define phrases after exact word-stream alignment.
Meter filtering reads the full `.mel`, including silent bars. Longer phrases are
rejected rather than split (`--max-phrase-notes`, default 40). Outputs are never
overwritten. Tempo is synthetic (`--tempo`, default 120 BPM); durations are
estimated up to the next syllabic onset/rest, capped at one local measure. This
creates bridge input and corpus annotations, not a prepared `bridge_eval` split:
RS stress is not bridge IPA supervision and vowel-length labels are absent. DALI
exclusion follows the supplied CSV; it does not establish absence from other
pretraining data.

## Development

```bash
conda activate prosodia-lyricist
python -m pytest -q
ruff check prosodia_lyricist tests
```

Tests use synthetic annotations and tiny local models. Real IPA integration tests
require installed pronunciation resources. Coverage includes global encoding,
pretraining-split audits, optional meter/contrastive fusion, label independence,
head gradients, padding, accumulation, checkpoint reload, MIDI tails/tempo/meter,
XAI import and causal scoring, and reference isolation.

```text
prosodia_lyricist/
  prepare.py, dali.py, ipa.py   DALI preparation and IPA supervision
  melody_encoder/             pretrained note representation and tower
  melody_checkpoint.py        strict trunk import and provenance
  bridge_data.py              global note features, targets, padding
  bridge_model.py              global encoder, optional fusion, direct note heads
  bridge_alignment.py         word-local IPA targets and observed note anchors
  meter.py                    observed meter features and missing-data masks
  bridge_train.py, train.py    supervised training and validation
  bridge_eval.py              held-out template evaluation and controls
  bridge_infer.py              MIDI-to-template prediction
  midi_eval.py                 MIDI/bridge template comparison on paired songs
  pipeline_eval.py             matched template-to-BART pipeline evaluation
  xai_features.py              XAI title/count/stress/length/remainder encoding
  xai_model.py                 original checkpoint import and lyric generation
  infer.py                    bridge-to-XAI generation CLI
  evaluation.py, report.py     lyric scoring and reports
configs/bridge.yaml            current workflow defaults
```

## Attribution and data sources

The initial code and compound prosody conditioning design came from
[**XAI-Lyricist: Improving the Singability of AI-Generated Lyrics with Prosody
Explanations**](https://www.ijcai.org/proceedings/2024/0872), Qihao Liang,
Xichu Ma, Finale Doshi-Velez, Brian Lim, and Ye Wang, IJCAI 2024,
pp. 7877–7885, DOI: 10.24963/ijcai.2024/872. The paper and supplementary
materials are retained in `docs/references/`.

DALI format and dataset information:
[official repository](https://github.com/gabolsgabs/DALI) and
[DALI v2 release](https://zenodo.org/records/3576083). Cite Gabriel
Meseguer-Brocal, Alice Cohen-Hadria, and Geoffroy Peeters, “DALI: a large
Dataset of synchronized Audio, LyrIcs and notes, automatically created using
teacher-student machine learning paradigm,” ISMIR 2018. Follow the dataset's
own license and terms when obtaining or using it; it is not redistributed here.
