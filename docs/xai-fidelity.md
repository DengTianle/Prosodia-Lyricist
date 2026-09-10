# XAI-Lyricist fidelity audit

This baseline is intended for controlled DALI experiments. It is closer to the
local XAI-Lyricist implementation after this revision, but it is **not an exact
scientific clone yet**. In particular, example context, prompts, pretrained
artifacts, and decoder correctness fixes below can affect results. Future
iterations should use the same prepared manifest and baseline configuration;
comparison with published XAI-Lyricist numbers requires resolving those differences.

The reference is the local `../XAI-Lyricist` checkout, specifically
`1_data_binarisation/binarise.py`, `0_build_dict/build_dictionary.py`,
`models/{conbart,melody_embedding,lyric_embedding,dataloader}.py`,
`3_train_bart/train_bart_con.py`, and `configs/configs.yaml`. This audit follows
active code as well as commented experiments; it does not assume every YAML
field was actually used. The checkout contains incomplete/commented paths and
references to external artifacts, so it is not an executable numerical oracle.

## Restored defaults

| Item | Local original | Current default |
| --- | --- | --- |
| Pronunciation | `prosodic.Text`, word and line syllables | Legacy `prosodic` word syllables, flattened in word order |
| Stress | Apostrophe: strong; backtick: substrong; otherwise weak | Same string rules |
| Length | IPA `ː`: Long; otherwise Short | Same string rule, independent of DALI duration |
| Syllable count | Pronunciation count, at most 40 per line | Same, even when DALI sung count differs |
| Normalization | Quote normalization, `cuz` → `cause`, hyphen → space, remove periods | Same in IPA path, after DALI whitespace normalization |
| Compound encoder | Concatenate BART token, length, remainder embeddings; linear projection | Same architecture |
| New feature embedding initialization | Normal, std `hidden_size**-0.5`, zero padding row | Restored |
| Embedding dropout | Declares 0.2; forward never calls it | Effective 0.0; configurable |
| Optimizer | AdamW, lr 5e-5, betas (0.9, 0.98), decay 0.001 | Same |
| Batch / epoch cap | 4 / 1000 | Same |
| Warmup | 2500 optimizer steps | Same |
| Schedule | Linear warmup helper with `num_training_steps=-1` | Literal same arguments under `xai_original` |
| Gradient clipping | None | None (`null`) |
| Batch order | Randomized once when building loader | Randomized once; seeded here |
| Epoch loss aggregation | Mean of batch losses | Same; token-weighted option retained |
| Early stopping | Configured patience 5 | Patience 5 on validation weighted-total loss |
| Sequence limits | 1024 encoder / decoder | 1024 / 1024, skip rather than truncate |
| Active baseline loss | Text only; auxiliary terms commented out | Text weight 1; auxiliary weights 0 |

The original scheduler call deserves special attention: with the standard
Transformers helper, warmup increases LR through step 2499, and LR becomes zero
at step 2500 and stays zero. The original trainer imports this helper from an
unavailable `hugtransformers` fork; its precise implementation cannot be verified
locally. The regression test verifies the standard helper's literal behavior,
not a claim about an unavailable fork. `linear` uses the configured finite epoch
budget; `constant_after_warmup` keeps peak LR after warmup. Both are explicit
alternative experiments. No finite schedule is silently substituted.

## Restored deterministic training scaffolding

Preparation stores a word list with pronunciation syllable counts in IPA mode.
Alternative lexical/unknown modes store DALI sung counts. This is deterministic
processing, without an LLM or generated pseudo-labels. Dataset tokenization
expands each word's attributes over its BPE tokens. Activating an auxiliary
weight requests the aligned labels, creates its output head, computes its
cross-entropy, and adds it to the text loss. Decoder hidden states carry gradients
from all enabled heads. Weights and head parameters survive checkpoint reload.
Text tokenization is identical with auxiliary terms enabled or disabled.

| Weight | Supervision |
| --- | --- |
| `word` | Next BART token |
| `syllable` | Number of syllables in the word represented by this BPE piece |
| `remainder` | Cumulative syllables remaining after that word |
| `sentence` | Class 1 for lyric tokens, class 0 for BOS/EOS in one-line examples |

All padding labels are -100. Syllable/remainder supervision ignores BOS/EOS.
Metrics include unweighted component losses and weighted total loss; training
and validation use the same weights. Example configuration:

```yaml
training:
  loss_weights:
    word: 1.0
    syllable: 1.0
    remainder: 1.0
    sentence: 0.0
```

The original commented code references sentence/syllable/remainder outputs
that the active BART wrapper does not return. It also mixes `lambda_syll`,
`lambda_syllable`, and absent `lambda_sent`. These new heads are a completion of
that intended pathway, not recovered pretrained heads or a claim of matching
an unpublished multitask implementation. Original target remainder code sets
`rem = line_syllable_num - num_syllables` independently for every word. Here it
decrements cumulatively; this is an explicit correctness deviation that only
affects remainder supervision. Sentence supervision is limited to boundaries
in the current one-line representation, not full-song phrase identity.

This work restores training scaffolding. Hard syllable constraints, word-by-word
pronunciation feedback during decoding, and MIDI evaluation are not implemented
or validated by these auxiliary objectives.

## DALI adapter differences

- Parent indices determine words and lines; note-text continuations and timing
  are validated. Original input was a serialized list of text/keyword samples.
- IPA templates use the lyric text, not DALI duration or sung count. The latter
  remain in `sung_syllables` for provenance. IPA syllables are not assigned
  invented timestamps when counts differ.
- English filtering existed and remains the default. `language: all` disables
  it; English pronunciation is still assumed by the IPA backend. Exact-ID
  allowlists are optional, intersect other filters, and do not set data splits.
- Grouped seeded 80/10/10 hash splitting replaces pre-existing train/valid
  text files. It prevents known artist/title/audio duplicates crossing splits.
- Invalid DALI alignments reject a line/song even in IPA mode. This is stricter
  than text-only preprocessing and can change corpus membership. Rejections,
  absent IDs, and selected IDs excluded by filters are recorded.
- JSONL/manifest schema replaces indexed pickle datasets. Schema version 2
  requires re-preparation; old prepared data is not silently reinterpreted.

## Remaining differences unrelated to changing the input dataset

| Area | Difference and consequence |
| --- | --- |
| Example context | Current examples are one line. Original text records can pack several lines under one title. Cross-line context, length distributions, batch token counts, and updates per song therefore differ. This remains a material limitation for a strict clone. |
| Prompts / separators | Current source uses title + syllable count + template + `<line>`; original adds per-line `<keywords>` and period separators. DALI does not supply original keyword fields, but omitting keyword extraction and retaining a new separator are still modeling differences. No AI keyword generator has been substituted. |
| Target formatting | Wordwise leading-space BPE encoding is restored for prepared records. Original appends a period per line; current targets end with EOS without that synthetic period. Synthetic/legacy records lacking `words` retain whole-text tokenization. |
| Pretrained artifacts | Current default loads public `facebook/bart-base` and expands its vocabulary. Original loads an external custom BART directory and then a private experiment checkpoint using positional shape-based key remapping. Those artifacts are unavailable at the configured paths. A local Hugging Face model directory can be specified here, but original wrapper checkpoints are not compatible. Starting weights are not proven equivalent. |
| Tokenizer / vocabulary | One expanded tokenizer is used for source and target here. Original uses a custom source tokenizer (asserts 50322 tokens) and a separate target tokenizer. Added-token IDs, embedding initialization, and softmax size may differ. |
| Feature table sizes | Source length IDs match (padding 0, long 1, short 2). Remainder ID is remaining+1. The current source table allocates max_syllables+1 entries; original allocates an additional unused maximum-remainder row. Auxiliary class tables are explicit numeric counts rather than original dictionary offsets and unused entries. |
| Decoder teacher forcing | Original passes unshifted target embeddings together with identical labels. Here BART shifts labels internally, avoiding access to the current prediction target. This is a deliberate correctness fix, not numerical equivalence. |
| Padding | Original attention masks are commented out and target padding is token 1. Here encoder padding is masked and target padding is excluded from CE. Loss values and gradients differ as a result. |
| Model configuration | Real training takes layer/head/FFN sizes and internal dropout from pretrained BART config. Original accepts `n_head=8`, `d_model=512`, `ffn_hidden=2048` in YAML but its wrapper does not rebuild the loaded BART with them. Copying those unused values would not reproduce its active architecture. |
| Randomness / batching | Seed 1234 is explicit; original mixes Python/NumPy shuffles without an equivalent recorded seed. Fixed batch order is restored, but exact permutations/RNG states are not reproduced. Worker count is 0 here versus original environment default 10. |
| Runtime / checkpoints | Modern stock Transformers/PyTorch, auto CUDA/MPS/CPU, strict state loading, atomic self-contained best checkpoints and JSON metrics replace the original fork/imports, CUDA assumptions, TensorBoard and permissive key remapping. Hardware and library versions can affect results. |
| Early stopping details | Strictly lower validation loss resets patience here; ties count as stale. The referenced original early-stopping implementation is absent as source in this checkout, so tie/delta behavior cannot be verified. |
| Pronunciation version | Original does not pin `prosodic`. This project pins legacy 1.6.2, but equivalence of its pronunciation dictionary to the original experiment is unproven. Its upstream build fails on Python 3.12 (`imp` removed). Both attempted 1.x installations failed in the requested environment; the subsequent source-download request was declined. IPA rules/integration wiring are tested with a substitute backend; the real-backend integration test is skipped while unavailable. |

## Validation and scope

Offline tests exercise IPA marker rules (including long vowels versus diphthongs),
IPA/DALI count disagreement, exact-ID selection independent of filenames,
English/unknown-language filtering, target alignment and ignored padding,
nonzero auxiliary gradients, combined loss, checkpoint round trips, and the
literal scheduler's zero-after-warmup behavior. Existing preparation, tiny-BART
training, and inference regression tests remain in place. The latter protect
existing behavior; no new MIDI inference work was undertaken.

These tests do not establish full-corpus IPA parsing, numerical agreement with
the original checkpoint, or a reproduction of the paper's results. Real IPA
preparation is blocked in the current Python 3.12 environment until the legacy
dependency can be installed compatibly. Resolve context/prompt/artifact
differences above before describing the system as differing *only* in dataset.
