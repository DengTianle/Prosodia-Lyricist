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
Earlier four-stream Prosodia checkpoints
and their explicit `configs/dali.yaml` training path remain supported; the
[paper implementation notes](docs/xai-fidelity.md) describe that older path.

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
`data.dali_dir` in [`configs/dali.yaml`](configs/dali.yaml). DALI JSON exports
with `info` and `annotations` fields are also supported. Paths in the YAML are
relative to the configuration file, so commands also work outside the repo.

```bash
prosodia-prepare --config configs/dali.yaml
```

This writes `data/dali/{train,valid,test}.jsonl`, `manifest.json`, and
`rejected.jsonl`. Each record is one complete song with an ordered `lines` list;
the manifest records configuration, pronunciation version, song/group split
assignments, song and line counts, and file checksums. Schema version 5 records
repo/paper IPA rules and pronunciation provenance. Repo-mode XAI training requires
new preparation; schema-4 paper data remains readable by earlier models.
Existing lyrics-only checkpoints can still perform legacy inference; they cannot
be converted into trained explainable models without retraining.
Preparation can be rerun and replaces only these generated outputs. Raw data
and old local experiment outputs are ignored by Git.

The default configuration selects English songs and assigns approximately
80%/10%/10% to train/validation/test using a seeded hash of song groups. All
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
prosodia-prepare --config configs/dali.yaml --english-only --song-ids-file song_ids.txt
```

CLI paths are relative to the current directory. Before training, save matching
values in the YAML: training checks its data settings against the manifest.
`language: all` disables language filtering; IPA pronunciation remains English.
`--limit` limits scanned files, so the resulting missing-ID list may be partial.

### What the model learns from DALI

- **Line and word membership:** DALI's parent indices link notes to words and
  words to lines. The target is the joined parent-word text.
- **Default IPA syllables, stress, and length:** `stress_source: ipa` parses
  lyric text with modern `prosodic`. Primary and secondary stress both map to
  `strong`; otherwise stress is `weak`. Long-vowel marks and English diphthongs
  map to `long`, following the paper and supplement. Counts use IPA syllables
  even when DALI's sung count differs. Unpronounceable words reject the song.
  Punctuation tokens, including question marks, are retained with zero syllables.
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
not DALI durations, matching the original text-derived template approach. Pitch
is not an input to this model, as in the original compound prosody embedding.
The optional sung mode remains dependent on DALI syllable and duration quality.

Invalid indices, orphan continuations, overlapping/out-of-bounds intervals,
empty words, and lines above `max_syllables` (default 40 per line) are reported.
Any unusable line rejects the whole song, preserving complete context without
joining across missing lines. At tokenization, songs exceeding 1024 source or
1024 target tokens (including title, prompts, separators and BOS/EOS) are
skipped and counted, never truncated or split into independent lines. Test data is held out from
training and checkpoint selection.

For a preparation smoke test, `--limit 100` scans the first 100 files. Use a
separate `prepared_dir` if you want to keep the complete prepared dataset;
small subsets may have empty splits, in which case training fails clearly.

## Train

```bash
# Two training and two validation batches; tiny random BART, real tokenizer/data.
prosodia-train --config configs/dali.yaml --smoke-test

# Fine-tune pretrained facebook/bart-base.
prosodia-train --config configs/dali.yaml
```

Training selects CUDA, then Apple MPS, then CPU when `device: auto`. Adjust
batch size for available memory. A new timestamped run directory under
`checkpoints/dali/` contains `run.json`, `metrics.jsonl`, and a self-contained
`best/` checkpoint with weights, model config, tokenizer, and feature metadata.
`--output-dir` specifies an alternative **new** run directory.
`--local-files-only` prevents Hugging Face downloads when models are cached.

The encoder receives `<title>…<sent_0><keywords>…<prosody>…<sent_1>…`,
with compound stress/length embeddings at each syllable. DALI has no keywords,
so keyword prompts are empty. The decoder produces **four aligned streams**:
lyrics, syllable count, stress, and length. Its input concatenates the embeddings
of all four previous symbols and projects them to BART's hidden size. All four
streams are shifted together for causal next-event prediction.

The paper uses lyric words as timesteps. Here, BART spells each word using BPE
pieces and then predicts a special `<word_end>` event, which carries that word's
three prosody labels. This avoids pronouncing incomplete BPE fragments. A word's
stress is strong if any syllable is stressed; its length is long if any syllable
is long. BOS/EOS, punctuation, and unfinished BPE pieces carry the auxiliary
`pad` class 0. Batch padding is `-100` and excluded from all four losses.

The default objective is:

```text
loss = CE_lyrics + CE_syllables + CE_stresses + CE_lengths
```

Each CE is averaged over valid target events, including non-word pad classes
in the three prosody vocabularies. `training.loss_weights` exposes `lyrics`,
`syllables`, `stresses`, and `lengths` for explicit ablations; all default to 1.
Metrics record each unweighted component and the weighted sum. All heads are
saved in the checkpoint. The default uses Adam with betas (0.9, 0.98), epsilon
1e-5, and a warmup followed by constant LR. The partial reference repository's
`xai_original` schedule remains available explicitly; it sets LR to zero after
warmup and is no longer the default.

`model.decoder_mode: lyrics` selects the previous experimental model, including
its old `word`/`syllable`/`remainder`/`sentence` loss names and encoder format.
Explainable training requires IPA-prepared data. Both modes preserve whole-song
context, enforce source/target token limits without truncation, and write strict,
self-contained checkpoints. Optimizer resume, distributed training, and mixed
precision are not implemented.

## End-to-end melody conditioning

This older joint soft-conditioning experiment remains available explicitly through
`configs/end_to_end.yaml`. The default workflow on this branch is the standalone
[supervised bridge](#learned-melody-to-template-bridge), trained with template CE
and joined to the unchanged XAI checkpoint only at inference.

[`configs/end_to_end.yaml`](configs/end_to_end.yaml) loads
`checkpoints/bestM2LCkpt.pt` directly. Its 272 state entries identify BART-base
dimensions (768 hidden units, 6 encoder/decoder layers, 3072 FFN units, 1024
positions, 50,322 vocabulary rows), a 3-class length embedding, a 42-class
remainder embedding, and a `2304 → 768` source projection. There are no trained
auxiliary decoder heads, decoder feature embeddings, or word-end events.
Import maps exact parameter names, verifies tied aliases, and loads strictly.

```text
melody → pretrained local windows → song encoder → parallel structural slot queries
       → soft stress/length mixtures + fixed per-line remainder embeddings
       → checkpoint's three-part source projection → BART encoder/decoder → lyric CE
```

Slot queries contain line, slot, and syllable-count embeddings. They attend to
the whole song and one another without a causal mask or prior prosody labels.
Softmax probabilities mix the checkpoint's source stress-token and length
embeddings before its original concatenation/projection. Remainder IDs count
down independently in every phrase (`count..1`, with `Remain_0` at ID 1).
Prompts follow the released MIDI/binary title/count/template/period format;
keyword markers and keyword text are omitted, even if present in the input record.
The same soft connection is used for training and generation. Lyric targets
contain plain BPE text, period phrase separators, and EOS. Generation starts
with BOS, as in reference inference; training shifts labels right with that
same BOS prefix. The public reference forward instead feeds unshifted target
embeddings while predicting those targets; that target leakage is not reproduced.
The only decoder objective is lyric CE. `bridge_anchor_weight` optionally adds
IPA supervision on intermediate source slots. Greedy/top-k generation has no
default repetition penalty, n-gram filter, forced BOS/EOS, or IPA correction.
The reference also terminates when BOS is sampled (its encoded EOS membership
test includes BOS); generation now preserves that stopping behavior.

`data.prosody_mode: actual` applies the repository rules during **preparation,
bridge training, MIDI inference, and evaluation**: primary/secondary/unstressed
remain `strong/substrong/weak`, and a syllable is long only if its IPA contains
`ː`. Diphthongs and ASCII colons alone do not mark length. The bridge has three
strength outputs and two length outputs, mixing all three pretrained stress
embeddings without collapsing secondary stress. New bridge checkpoints use
format 7; earlier binary checkpoints retain their saved paper rules.

Pronunciation uses the same pinned Prosodic 1.5.0 English dictionary order and
`maybestressed`/`unstressed` precedence as the baseline branch. Modern Prosodic
provides tokenization and explicit dictionary-miss fallback; reports record
fallback words. The authors' exact historical Prosodic version is unrecorded.
Install the verified data once, then regenerate the new `data/dali-xai` dataset
before training (the old `data/dali-bridge` labels are incompatible):

```bash
python -m prosodia_lyricist.historical_ipa
python -m prosodia_lyricist.prepare --config configs/end_to_end.yaml
```

The audit also aligns MIDI stress with the released duration-dependent floor
grid using the **file's ticks per beat**, including non-480 files; duration
thresholds use onset buckets of eight quarter notes, computed before phrase
filtering. Only marked phrases enter the repo input. Notes starting after the
final marker are omitted and counted in diagnostics, matching `mid2words`.
The bundled Imagine input therefore uses 16 phrases/103 notes; its 10 trailing
notes still contribute to `getProsody` bucket thresholds. Paper/supplement
parsing remains an explicit `--prosody-mode paper` comparison.
To score all 17 lines in `examples/Imagine_gt.txt` with a joint XAI checkpoint,
add a phrase-end marker at the last note's end in a copy of `examples/imagine.mid`.

Training joins notes and lyrics from each prepared song, uses reference IPA counts,
and skips whole songs exceeding either model's limits. MIDI inference uses note
counts; neither path infers melisma alignment. Set `melody_checkpoint` and
`pretraining_manifest` to your contrastive pretraining assets. The default keeps
the melody tower frozen and adapts the bridge plus all of BART from epoch 0.
Set `template_unfreeze_epoch: null` to train only the bridge, or use
`template_finetune: encoder` to adapt the BART source encoder/projection and its
source-only length/remainder embeddings while freezing decoder/shared token
weights. `melody_unfreeze_epoch: 0` also adapts the melody tower. Component
learning rates and later unfreeze epochs remain configurable.

**Missing author assets:** the public repository and its fork do not include
`bart_tokenizer` or `bart_custom_model/config.json`. By default, the tokenizer
recovers title/template/stress IDs from the bundled tokenized samples and
extends their observed count-5/6/7 mapping linearly to counts 1..40; unknown,
unused added IDs keep opaque names. The original checkpoint's use of this
sample vocabulary is unverified; the `counter` examples have a different count
mapping. Head counts (12), activations, and dropout
use BART-base settings, which cannot be established from weights alone.
These assumptions are saved in import and run provenance. Set
`model.xai_tokenizer_dir` / `model.xai_bart_config` to the author's files when
available. Matching tensor shapes establishes architecture compatibility,
not an exact numerical reproduction of the author's unavailable setup.

```bash
python -m prosodia_lyricist.train --config configs/end_to_end.yaml --smoke-test
python -m prosodia_lyricist.train --config configs/end_to_end.yaml
python -m prosodia_lyricist.infer --checkpoint checkpoints/end-to-end-xai/RUN/best \
  --midi examples/imagine.mid --title Imagine --report-prefix outputs/joint-imagine
python -m prosodia_lyricist.end_to_end_eval --checkpoint checkpoints/end-to-end-xai/RUN/best \
  --prepared-dir data/dali-xai --device cuda --precision bf16 \
  --generation-limit 16 --compare-hard --melody-controls --output outputs/joint-eval.json
```

To generate with the original checkpoint and heuristic MIDI templates before
training a melody bridge, import it once into a new self-contained directory:

```bash
python -m prosodia_lyricist.xai_model --checkpoint checkpoints/bestM2LCkpt.pt \
  --output-dir checkpoints/xai-original --local-files-only
python -m prosodia_lyricist.infer --checkpoint checkpoints/xai-original \
  --midi examples/imagine.mid --title Imagine --top-k 3
```

Omit `--local-files-only` to download the base BART tokenizer if it is not cached.
Import also accepts `--tokenizer-dir` and `--bart-config` for original assets.
Joint checkpoints include both models and the tokenizer and reload without the
original `.pt`, reference checkout, or contrastive checkpoint.

Smoke tests use the actual configured pretrained weights, with two small training
batches; they do not silently substitute random pretrained models. `train.sh`
defaults to this configuration and accepts another config as its first argument.
Validation selects checkpoints and drives early stopping by the combined weighted
loss (lyric CE plus the optional bridge anchor), and displays that
same combined score in epoch summaries. It logs lyric loss and bridge accuracy.
There are no auxiliary decoder accuracy metrics for the single-stream model.
The separate evaluator measures IPA from
actual generated lyrics, includes duration baselines on the same generation songs,
and optionally compares hard embeddings and wrong melodies on matched subsets.
Use `--count-source notes` for the note-count deployment condition; the default
`ipa` isolates conditioning with reference counts. Reference agreement and agreement
with the bridge's hard preview are reported separately; neither establishes physical
singability. Run metadata records both pretraining sources and split-verification
status; different template-pretraining data hashes leave that overlap unverified.
To run the previous four-stream experiment, point `template_checkpoint` to its
template-only checkpoint, set `decoder_mode: explainable`, and restore its four
`loss_weights` (`lyrics`, `syllables`, `stresses`, `lengths`).

## Learned melody-to-template bridge

The bridge uses the window-preserving melody encoder and global song features
from `prosodia-bridge2`. Two bidirectional song-encoder layers provide whole-song
context. Each requested syllable position then retrieves from that memory using
four cross-attention/FFN layers, with independent stress and vowel-length heads:

```text
MIDI notes → pretrained two-line windows → whole-song encoder
           → parallel position queries → discrete IPA template
           → original XAI-Lyricist template encoder and lyric decoder → lyrics
```

Queries contain only line, within-line position, count, and sinusoidal position
embeddings. There are no previous-label inputs, shifted labels, causal masks,
query self-attention, or decoding loops. Both supervised losses backpropagate
through the readout into the encoder when it is unfrozen:

```text
loss = CE_stress + CE_length
```

The default `data.prosody_mode: actual` preserves XAI's three stress classes
(`strong`, `substrong`, `weak`) and binary vowel length, where only the IPA
long mark means long. Prefixes and padding contribute no loss. The title and
lyric text never enter the bridge; the title is supplied to XAI at inference.
The XAI checkpoint is neither loaded nor updated during bridge training.

Training and evaluation use reference IPA syllable counts to request positions.
MIDI inference uses one position per note and retains an unmarked tail as a final
phrase. There is no count predictor or forced note-to-syllable alignment; melisma
can still make training and inference counts differ.

Configure `configs/bridge.yaml` with the original contrastive checkpoint and its
pretraining CSV. Set those paths before preparation so identity groups exposed to
upstream training stay in downstream training. Other duplicate song/audio groups
also stay together. Source limits remain 512 notes per encoder window, 2,048 notes
and 256 lines per song, and 4,096 query/structural positions per song. Overlong
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
matches. Old binary paper-mode `data/dali-bridge` labels cannot supervise an XAI
bridge. For upstream CSV changes, `training.pretraining_audit_mode: warning`
retains prepared-file integrity checks and records missing/conflicting audit
information; `strict` requires the unchanged original CSV.

Evaluation schema v4 reports one `prediction` block: slot-weighted loss, separate
loss components, head/pair accuracies, exact phrase accuracy, prosody-BLEU, and
note/IPA count diagnostics. All are computed from the same parallel pass, without
gold-label feedback. Unsmoothed BLEU-4 scores phrases shorter than four slots as
zero. Duration baselines, constant controls, and the equal-note/IPA-count subset
are retained. Checkpoint selection uses this same validation CE.

New checkpoints use bridge format v8. Older v2-v7 checkpoints retain their original
architecture for inference/evaluation; they do not initialize the new readout.
See [bridge details](docs/prosody-bridge.md) for features and configuration.

## MIDI inference

MIDI remains the inference input. All phrases are encoded together under one
title and decoded with one shared song context, giving the same song-level context as training.
Generated periods become output line breaks. The model learns boundaries but
does not enforce the requested number of lines. The default generation budget
is the checkpoint target limit minus one decoder-start token (1023 for the
standard configuration); `--max-new-tokens` can lower it. Overlong source songs
raise an error without truncation:


```bash
prosodia-infer \
  --checkpoint checkpoints/dali/RUN_ID/best \
  --midi examples/imagine.mid \
  --title Imagine \
  --top-k 1 \
  --output outputs/imagine.txt \
  --explanations outputs/imagine.prosody.json \
  --report-prefix outputs/imagine
```

Choose a monophonic melody with `--track` (default 0). Markers are interpreted
as **phrase ends**, matching the supplied example; they are not beat markers.
A trailing phrase after the final marker is retained in paper/supplement mode;
repo (`actual`) mode uses only marked phrases. Each note is assumed to
represent one syllable at this stage. MIDI melisma and explicit beat/bar marker
support remain future evaluation work.

Default `--stress-source supplement` matches the baseline evaluation's explicit
reconstruction: duration-dependent 4/4 strength, nearest note-type-grid rounding,
and melody-mean duration. The [evaluation guide](docs/evaluation.md) records the
supplement's ambiguities and exact conventions. Tick zero anchors the grid;
explicit non-4/4 meters are rejected. `heuristic` preserves the historical
unquantized strength/phrase-mean behavior; `unknown` disables stress.

Imagine has 113 notes and 16 phrase markers plus an unmarked tail, yielding
**17 phrases in one model input** in paper/supplement mode, or 16 marked phrases
in repo mode. A trained template-decoder checkpoint is
needed for lyrics. Inspect the musical template without any checkpoint:

```bash
python -m prosodia_lyricist.infer --midi examples/imagine.mid --title Imagine \
  --template-only --report-prefix outputs/imagine-template-decoder-template
```

`--report-prefix PREFIX` writes `PREFIX.md` and `PREFIX.json`: musical and
generated templates, IPA, note positions, predicted/corrected decoder features,
phrase counts, prosody-BLEU-4 with n-gram diagnostics, and conditional perplexity.
`--reference lyrics.txt` adds reference scoring (one nonempty line per input
phrase), without giving those words to generation. Generated-text PPL is a
self-score. Explainable scoring replays all four causal streams, excludes
auxiliary losses, and reports lyric-BPE and all-event PPL separately. The
extra decoder history limits direct PPL comparison with baseline. Prosody-BLEU
uses the same independent extractor and formula in both branches; see the
guide for limitations relative to the paper. Output paths must be new.

`--top-k 1` uses greedy decoding; larger values use top-k temperature sampling.
The four-stream generator honors the checkpoint's `no_repeat_ngram_size` and
`repetition_penalty`. These controls act on lyric BPE tokens and punctuation;
word-end events and auxiliary feature streams are excluded. Override them with
`--no-repeat-ngram-size N` and `--repetition-penalty P`; use `0` and `1` respectively
to disable them. Effective settings are recorded in reports and explanations.
At each `<word_end>`, all three prosody heads sample a non-pad label. The IPA
parser recomputes the completed word's labels, replaces disagreements, and
feeds the corrected compound event into the cached decoder **before** the next
word. Missing pronunciation fails explicitly. `--no-prosody-correction` is an
ablation that instead feeds back sampled labels without calling IPA during
generation; reporting still extracts IPA independently for evaluation.

`--explanations` writes the four aligned streams, readable token strings,
per-word sampled/corrected labels, full IPA syllables, source melody features,
and completion status. Python callers can use `infer(..., return_explanations=True)`
or `model.generate(..., tokenizer=tokenizer)` for structured results. The text
output contains only lyrics. If the token budget cuts a word short, that unfinished
word is excluded and reported as `truncated_word` in the JSON. Generation remains
softly conditioned: it does not enforce the requested number of syllables or lines.

## Development

```bash
python -m pytest -q
ruff check prosodia_lyricist tests
ruff format --check prosodia_lyricist tests
```

Tests use synthetic annotations and a tiny local tokenizer/model; no DALI
files or model downloads are needed. Real IPA integration tests run when
`prosodic` is installed and require its tokenizer/pronunciation resources. They cover IPA feature rules with a test backend, four-stream gradients, causal shifting, correction feedback, and
reload, song packing, source/target period boundaries, exact 1024-token limits,
whole-song rejection, modern IPA pronunciation selection, original scheduler behavior, exact-ID/language filtering, parent alignment, melisma,
invalid annotations, lexical stress, split reproducibility, padding, shifted
labels, optimization, checkpoint reload/generation, and MIDI phrase handling.

```text
prosodia_lyricist/
  dali.py          DALI reader and aligned prosody extraction
  prepare.py       JSONL preparation, splits, and provenance
  ipa.py           paper IPA rules and deterministic pronunciation
  features.py      shared template encoding
  data.py          dataset validation and batch padding
  model.py         four-stream compound embeddings, losses, and checkpoint IO
  generation.py    word completion, prosody sampling, and IPA correction
  legacy_*.py      previous lyrics-only experimental model and encoding
  train.py         training and validation
  midi.py          MIDI phrase/prosody conversion
  infer.py         MIDI generation CLI
  evaluation.py    independent prosody-BLEU and causal conditional perplexity
  report.py        Markdown/JSON template and lyric evaluation reports
  config.py        configuration loading
  runtime.py       seeds and device selection
configs/dali.yaml  project defaults
examples/          phrase-marked MIDI example
tests/            offline regression tests
docs/references/  original research papers
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
