# Prosodia Lyricist

Prosody-conditioned lyric generation with BART, trained on the **DALI music
annotation dataset**. The intended baseline follows XAI-Lyricist with a DALI dataset adapter.
It is not yet an exact reproduction: see the [fidelity audit](docs/xai-fidelity.md)
for restored settings, deliberate correctness fixes, and outstanding differences.
Old checkpoints, dictionaries, and training formats are not supported.

## Setup

Use the existing Conda environment (its name on this machine is
`prosodia-lyricist`):

```bash
conda activate prosodia-lyricist
python -m pip install -e '.[midi,dev]'
# IPA mode also needs the legacy pronunciation package; see compatibility note below.
python -m pip install -e '.[ipa]'
# Optional: keep downloaded Hugging Face files inside this project.
export HF_HOME="$PWD/data/huggingface"
```

Python 3.10+ is required. `midi` adds MIDI inference; `dev` adds pytest and Ruff.
Preparation needs no audio downloads, MIDI files, DALI Python package, or custom
Transformers fork. Default IPA preparation uses `prosodic==1.6.2`, retaining the
original 1.x `Text.words()` / `word.syllables()` API. Its upstream installer uses
`imp`, removed in Python 3.12: the `ipa` extra currently needs Python 3.10/3.11
or a separately validated compatibility patch. Installation in this machine’s
Python 3.12 `prosodia-lyricist` environment failed; real IPA integration remains
unverified there. The lexical/unknown alternatives do not need this dependency. The first training run
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
  lyric text with legacy `prosodic`, as in XAI-Lyricist. An apostrophe in the
  syllable string means `strong`, a backtick means `substrong`, otherwise `weak`.
  The IPA length mark `ː` means `long`, otherwise `short`. Counts and remainders
  use IPA syllables even when DALI's sung count differs. Unpronounceable words
  reject the line rather than silently substituting another parser.
- **Optional sung features:** `stress_source: lexical` retains the previous
  DALI sung-count / CMUdict stress / relative-duration length mode.
  `stress_source: unknown` uses sung counts and duration with unknown stress.
  Empty or `~` continuation notes merge into the preceding syllable. Durations
  sum sounding intervals without gaps; above-line-mean durations are long.
- **Alignment provenance:** `sung_syllables` retains DALI note timing separately
  from the IPA template; `words` retains deterministic per-word syllable counts.
- **Remainder:** each syllable carries the number of syllables remaining in its
  line, with a separate padding ID.

**DALI has no bar-line annotations.** No tempo, meter, downbeats, or bar lines
are inferred from its annotation frame rate. Pronunciation stress is a lexical
proxy, not measured musical stress. Default training uses IPA vowel length,
not DALI durations, matching the original text-derived template approach. Pitch
is not an input to this model, as in the original compound prosody embedding.
The optional sung mode remains dependent on DALI syllable and duration quality.

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
masked and target padding uses `-100`, so it contributes no loss. The default settings now match the original active training choices: AdamW,
learning rate 5e-5, betas (0.9, 0.98), weight decay 0.001, batch size 4,
1000 maximum epochs, 2500 warmup steps, no gradient clipping, batch-mean
validation loss, fixed batch order, and patience 5. Embedding dropout is zero
because the original defines but does not apply it; BART's internal dropout
retains its pretrained configuration.

**The default `schedule: xai_original` reproduces the literal original
`num_training_steps=-1`: learning rate becomes zero at step 2500.** Choose
`constant_after_warmup` or `linear` explicitly to run a different schedule.
These are experimental deviations, not silently substituted defaults.

`training.loss_weights` controls `word`, `syllable`, `remainder`, and `sentence`
losses. Non-text weights default to zero, matching the active original baseline.
For example, `syllable: 1.0` and `remainder: 1.0` activate deterministic word/BPE
label scaffolding and trainable auxiliary heads. Each word's BPE pieces share
its syllable count and the cumulative number of syllables remaining after the
word. Sentence supervision distinguishes lyric tokens from BOS/EOS in the
current single-line examples. Padding is ignored by every loss. Metrics record
unweighted component losses and their weighted total; checkpoints save active
heads and weights. This completes the original commented training scaffolding;
it does not implement constrained MIDI decoding. Loading is strict; inference reconstructs
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
four-beat heuristic, while default DALI training uses IPA stress and vowel length. This is a known
conditioning mismatch to revisit when beat markers are available. Pass
`--stress-source unknown` to disable the heuristic; this is the automatic
choice for checkpoints trained with `stress_source: unknown`. Duration uses
the relative phrase-mean rule of the optional sung training mode; this differs
from default IPA vowel length. `--top-k 1` uses greedy
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
files or network connection are needed. They cover IPA feature rules with a test backend, auxiliary-head gradients and
reload, original scheduler behavior, exact-ID/language filtering, parent alignment, melisma,
invalid annotations, lexical stress, split reproducibility, padding, shifted
labels, optimization, checkpoint reload/generation, and MIDI phrase handling.

```text
prosodia_lyricist/
  dali.py          DALI reader and aligned prosody extraction
  prepare.py       JSONL preparation, splits, and provenance
  ipa.py           original IPA pronunciation and deterministic word scaffolding
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
