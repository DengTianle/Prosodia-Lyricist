# Prosodia Lyricist

Prosody-conditioned lyric generation with BART, trained on the **DALI music
annotation dataset**. This is a new project derived from XAI-Lyricist, rather
than the official implementation or a reproduction of its published results.
Old checkpoints, dictionaries, and training formats are not supported.

## Setup

Use the existing Conda environment (its name on this machine is
`prosodia-lyricist`):

```bash
conda activate prosodia-lyricist
python -m pip install -e '.[midi,dev]'
# Optional: keep downloaded Hugging Face files inside this project.
export HF_HOME="$PWD/data/huggingface"
```

Python 3.10+ is required. `midi` adds MIDI inference; `dev` adds pytest and Ruff.
Preparation needs no audio downloads, MIDI files, DALI Python package, external
pronunciation service, or custom Transformers fork. The first training run
fetches the BART tokenizer and pretrained weights from Hugging Face.

## Prepare DALI

Place the official horizontal `.gz` annotation files in `DALI_v2/`, or set
`data.dali_dir` in [`configs/dali.yaml`](configs/dali.yaml). DALI JSON exports
with `info` and `annotations` fields are also supported. Paths in the YAML are
relative to the configuration file, so commands also work outside the repo.

```bash
prosodia-prepare --config configs/dali.yaml
```

This writes `data/dali/{train,valid,test}.jsonl`, `manifest.json`, and
`rejected.jsonl`. Each record is one lyric line; the manifest records the
configuration, song/group split assignments, counts, and file checksums.
Preparation can be rerun and replaces only these generated outputs. Raw data
and old local experiment outputs are ignored by Git.

The default configuration selects English songs and assigns approximately
80%/10%/10% to train/validation/test using a seeded hash of song groups. All
lines from a song stay together. Entries sharing normalized artist/title or
the same audio ID are grouped together, including transitive duplicates.
This does not identify every cover or near-duplicate recording in DALI.
`min_ncc` optionally filters DALI's alignment score; the default 0.0 adds no
quality threshold. Unknown-language songs are excluded by the English filter.

### What the model learns from DALI

- **Line and word membership:** DALI's parent indices link notes to words and
  words to lines. The target is the joined parent-word text.
- **Sung syllables:** note-text segments define syllables. Empty or `~`-prefixed
  continuation notes merge into the preceding syllable within the same word.
  This is annotation-derived, not an assumption that every note is a syllable.
- **Length:** sum each syllable's note durations, excluding gaps. A duration
  above its line's mean is `long`; otherwise it is `short`.
- **Stress:** CMUdict lexical stress supplies `strong`, `substrong`, or `weak`
  only when a pronunciation's syllable count matches DALI's sung count.
  Missing pronunciations or count mismatches become `unknown`. Set
  `stress_source: unknown` to use duration/count conditioning alone.
- **Remainder:** each syllable carries the number of syllables remaining in its
  line, with a separate padding ID.

**DALI has no bar-line annotations.** No tempo, meter, downbeats, or bar lines
are inferred from its annotation frame rate. Lexical stress is a pronunciation
proxy, not measured musical stress. This retains the earlier project's
prosody-template idea while using DALI's alignments and sung durations. Pitch
is not an input to this model, as in the original compound prosody embedding.
Syllable and duration labels remain dependent on DALI annotation quality.

Invalid indices, orphan continuations, overlapping/out-of-bounds intervals,
empty words, and lines above `max_syllables` are reported and rejected. A line
error discards that line; broken parent indices discard the song because its
membership is ambiguous. Long token sequences are subsequently skipped with
counts at training time, never silently truncated. Test data is held out from
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

The encoder projects concatenated token, length, and remainder embeddings.
BART shifts decoder labels for next-token prediction. Encoder padding is
masked and target padding uses `-100`, so it contributes no loss. Training uses
AdamW, a finite warmup/linear schedule, gradient clipping, token-weighted
validation loss, and early stopping. Loading is strict; inference reconstructs
the model entirely from its checkpoint without fetching the base model.
Optimizer resume, distributed training, mixed precision, and full-song context
are not implemented. Each training example and generated MIDI phrase is one
independent lyric line.

## MIDI inference

MIDI remains the inference input for now:

```bash
prosodia-infer \
  --checkpoint checkpoints/dali/RUN_ID/best \
  --midi examples/imagine.mid \
  --title Imagine \
  --output outputs/imagine.txt
```

Choose a monophonic melody with `--track` (default 0). Markers are interpreted
as **phrase ends**, matching the supplied example; they are not beat markers.
A trailing phrase after the final marker is retained. Each note is assumed to
represent one syllable at this stage. MIDI melisma and explicit beat/bar marker
support remain future evaluation work.

The default MIDI stress feature uses the earlier project's duration-dependent
four-beat heuristic, while DALI training uses lexical stress. This is a known
conditioning mismatch to revisit when beat markers are available. Pass
`--stress-source unknown` to disable the heuristic; this is the automatic
choice for checkpoints trained with `stress_source: unknown`. Duration uses
the same relative phrase-mean rule as training. `--top-k 1` uses greedy
prediction; larger values sample with `--temperature`. Generation is
conditioned on prosody but does not enforce a hard syllable-count constraint.
The old experimental parody and saliency scripts have been removed.

## Development

```bash
python -m pytest -q
ruff check prosodia_lyricist tests
ruff format --check prosodia_lyricist tests
```

Tests use synthetic annotations and a tiny local tokenizer/model; no DALI
files or network connection are needed. They cover parent alignment, melisma,
invalid annotations, lexical stress, split reproducibility, padding, shifted
labels, optimization, checkpoint reload/generation, and MIDI phrase handling.

```text
prosodia_lyricist/
  dali.py          DALI reader and aligned prosody extraction
  prepare.py       JSONL preparation, splits, and provenance
  features.py      shared template encoding
  data.py          dataset validation and batch padding
  model.py         compound embeddings and BART checkpoint IO
  train.py         training and validation
  midi.py          MIDI phrase/prosody conversion
  infer.py         MIDI generation CLI
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
