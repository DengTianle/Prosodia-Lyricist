# Prosodia Lyricist

Prosody-conditioned lyric generation with BART, trained on the **DALI music
annotation dataset**. This branch trains a supervised parallel bridge from melody
to IPA templates, then feeds its discrete predictions to the original
XAI-Lyricist `bestM2LCkpt.pt`. Each requested syllable position attends to the
whole melody encoding and predicts stress and vowel length with separate CE
losses. The bridge has no label feedback or autoregressive decoding. `train.sh`,
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
approximately 80%/10%/10% train/validation/test song groups. Groups exposed to
contrastive pretraining's training split are then assigned to downstream train. All
lines from a song stay together. Entries sharing normalized artist/title or
the same audio ID are grouped together, including transitive duplicates.
This does not identify every cover or near-duplicate recording in DALI.
`min_ncc` optionally filters DALI's alignment score; the default 0.0 adds no
quality threshold. Unknown-language songs are excluded by the English filter.

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

The bridge uses the window-preserving melody encoder and global song features
from `prosodia-bridge2`. Two bidirectional song-encoder layers provide whole-song
context. Shared stress and vowel-length classifiers read each contextual note
directly after those bidirectional layers:

```text
MIDI notes → pretrained two-line windows → whole-song encoder
           → per-note stress/length heads → discrete IPA template
           → original XAI-Lyricist template encoder and lyric decoder → lyrics
```

Observed DALI word/note links map the first note of each sung-syllable group to
an IPA slot. Gold labels and this mapping select losses and scores; they never
enter the encoder or classification heads. Both supervised losses backpropagate
through the song encoder and, when unfrozen, the melody tower:

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

Configure `configs/bridge.yaml` with the original contrastive checkpoint and its
pretraining CSV. Set those paths before preparation so identity groups exposed to
upstream training stay in downstream training. Other duplicate song/audio groups
also stay together. Source limits remain 512 notes per encoder window, 2,048 notes
and 256 lines per song, and 4,096 IPA/structural positions per song. Overlong
training songs are skipped whole and counted; inference rejects them.

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

Evaluation schema v5 reports one `prediction` block: slot-weighted loss, separate
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

New checkpoints use bridge format v9. Older v2-v8 checkpoints retain their original
architecture for inference/evaluation. New training defaults to note heads;
explicit legacy parallel-query configurations remain usable for comparisons.
See [bridge details](docs/prosody-bridge.md) for features and configuration.

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

## Development

```bash
conda activate prosodia-lyricist
python -m pytest -q
ruff check prosodia_lyricist tests
```

Tests use synthetic annotations and tiny local models. Real IPA integration tests
require installed pronunciation resources. Coverage includes song packing,
pretraining-split audits, global feature fusion, parallel label independence,
head gradients, padding, accumulation, checkpoint reload, MIDI tails/tempo/meter,
XAI import and causal scoring, and reference isolation.

```text
prosodia_lyricist/
  prepare.py, dali.py, ipa.py   DALI preparation and IPA supervision
  melody_encoder/             pretrained note representation and tower
  melody_checkpoint.py        strict trunk import and provenance
  bridge_data.py              windows, global song features, targets, padding
  bridge_model.py              source fusion, song encoder, direct note heads
  bridge_alignment.py         word-local IPA targets and observed note anchors
  bridge_readout.py           legacy cross-attention query readout
  bridge_train.py, train.py    supervised training and validation
  bridge_eval.py              held-out template evaluation and controls
  bridge_infer.py              MIDI-to-template prediction
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
