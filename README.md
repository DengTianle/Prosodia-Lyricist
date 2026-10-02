# Prosodia Lyricist

Explainable prosody-conditioned lyric generation with BART, trained on the
**DALI music annotation dataset**. The default implements XAI-Lyricist's four
aligned decoder streams, compound decoder feedback, equal-weight four-part
cross-entropy loss, and IPA prosody correction before the next word is generated.
See the [paper implementation notes](docs/xai-fidelity.md) for the BART subword
adaptation and differences from the published experiment. Previous song-level
lyrics-only checkpoints remain loadable through an explicit legacy path.

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

The adapter selects the first pronunciation for each word, preserves repeated
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
assignments, song and line counts, and file checksums. Schema version 4 requires
re-running preparation and training; older prepared data must be rebuilt because the IPA stress/length rules changed.
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

## Learned melody-to-template bridge

The bridge trains the melody encoder from the `prosodia-direct` contrastive
integration to predict the **syllable-level IPA templates** consumed by the
existing template decoder:

```text
MIDI notes → contrastively pretrained melody encoder → learned prosody template
           → existing template-decoder checkpoint → lyrics
```

It fills a fixed line skeleton with stress/vowel-length pairs. DALI line
membership and lyric-derived IPA counts define the training skeleton. The only
training loss is cross-entropy on prosody pairs; there is no count-prediction head
or loss. At inference, MIDI markers define the phrases and each note supplies
exactly one prosody slot. The decoder does not generate line endings. All phrase
templates are passed to the lyric model together, preserving whole-song context.

In [`configs/bridge.yaml`](configs/bridge.yaml), set `model.melody_checkpoint`
to the current note-based two-pool contrastive checkpoint and
`data.pretraining_manifest` to its original CSV **before preparation**. Match
`data.lines_per_window` to that manifest. Preparation uses `data/dali-bridge`
and keeps shared songs and duplicate identities in their pretraining splits.

```bash
conda activate prosodia-lyricist
python -m prosodia_lyricist.prepare --config configs/bridge.yaml
python -m prosodia_lyricist.train --config configs/bridge.yaml --smoke-test
python -m prosodia_lyricist.train --config configs/bridge.yaml

python -m prosodia_lyricist.infer \
  --checkpoint checkpoints/dali/TEMPLATE_RUN/best \
  --bridge-checkpoint checkpoints/bridge/BRIDGE_RUN/best \
  --midi examples/imagine.mid --title Imagine --top-k 1 \
  --report-prefix outputs/imagine-bridge
```

`--bridge-checkpoint` replaces heuristic MIDI template construction. Omit it to
retain existing inference. Use it with `--template-only` to inspect predictions
without loading the lyric model. Reports retain both the predicted syllable
templates and original note arrays; reference lyrics are used only for scoring.
No skeleton JSON is needed; `--bridge-skeleton` has been removed. Phrase note
counts must fit `max_syllables`; overlong input fails without truncation. Generation
validation reports skipped overlong windows and note/IPA count differences as data
diagnostics, rather than learned count accuracy. Reports also list the raw notes,
bar:beat positions, and old beat-based labels as a separate comparison.
The v3 bridge can load v2 scaffold checkpoints, discarding their obsolete count
head; v1 LINE_END checkpoints still require retraining. Existing prepared IPA data
and the template-decoder checkpoint can be reused. Training IPA counts can differ
from inference note counts because of melisma; exact slot counts do not guarantee
that the downstream lyric model produces the same number of syllables.

Evaluate a bridge directly on the prepared held-out `test` split (or use
`--split valid`). This loads the checkpoint's limits without needing the training YAML:

```bash
python -m prosodia_lyricist.bridge_eval \
  --checkpoint checkpoints/bridge/BRIDGE_RUN/best \
  --prepared-dir data/dali-bridge --device cuda --precision bf16 \
  --output outputs/bridge-eval.json
```

The JSON reports slot-weighted cross-entropy and strength, length, and combined
accuracy with teacher forcing on the IPA skeleton. A separate greedy evaluation
uses note counts and reports the same accuracies by slot order (missing/extra
slots count as incorrect), exact phrase accuracy, and the existing phrase-mean
prosody-BLEU. This is unsmoothed BLEU-4, so phrases shorter than four slots score
zero even when correct. Count mismatches and skipped windows are also reported.
Use `--limit 100` for a quick subset check; omit it for the full split. On CPU/MPS,
use `--precision fp32` (the default).

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
A trailing phrase after the final marker is retained. Each note is assumed to
represent one syllable at this stage. MIDI melisma and explicit beat/bar marker
support remain future evaluation work.

Default `--stress-source supplement` matches the baseline evaluation's explicit
reconstruction: duration-dependent 4/4 strength, nearest note-type-grid rounding,
and melody-mean duration. The [evaluation guide](docs/evaluation.md) records the
supplement's ambiguities and exact conventions. Tick zero anchors the grid;
explicit non-4/4 meters are rejected. `heuristic` preserves the historical
unquantized strength/phrase-mean behavior; `unknown` disables stress.

Imagine has 113 notes and 16 phrase markers plus an unmarked tail, yielding
**17 phrases in one model input**. A trained template-decoder checkpoint is
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
  report.py        Markdown/JSON template, lyrics, and decoder-feedback reports
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
