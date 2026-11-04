# ProsodyOctuple for the Lakh bridge

Implementation specification, 2026-11-04. Representation ID:
`prosody_octuple_v1`. Train a new melody-to-IPA bridge from scratch on the
calibrated Melistas/Lakh subset, then use the existing XAI template decoder.
The first experiment has no contrastive encoder or self-supervised pretraining.
Use `configs/bridge_lakh_octuple.yaml` for the encoding comparison and
`configs/bridge_lakh_octuple_roles.yaml` for learned syllable starts. The matched
177-D baseline is `configs/bridge_lakh_canonical.yaml`; all share `data/lakh-octuple`.
Preparation reuses verified IPA targets and split membership from the base dataset.

Borrow OctupleMIDI's **one compound token per note**, independent field
embeddings, and concatenation/projection into a Transformer. Replace instrument
and velocity with melodic interval and preceding silence. The source export
did not retain velocity; its exported MIDI uses constant 80. Instrument identity
adds little to this selected vocal melody input. This is a new representation,
not compatible with MusicBERT weights. The current 177-D encoder already uses
one token per note, so this changes information and inductive bias, not sequence
length. [MusicBERT paper](https://aclanthology.org/2021.findings-acl.70.pdf),
[official codec](https://github.com/microsoft/muzic/blob/main/musicbert/preprocess.py).

The local prepared manifest contains 3,529 usable songs, 1,048,290 notes, and
1,012,088 supervised IPA slots out of 1,028,245 (98.43%). Use
`data/leakage-clean-20261103/lakh-bridge`, whose train/valid/test counts are
2,785/364/380, as the starting membership. The raw source is
`outputs/lakh-alignment-audit/deduplicated-orthographic.jsonl.gz`.
Read-only training profiling found median 283 notes/song, maximum 1,058 notes,
2.34% continuation notes, and a maximum span of 279 bars. It also found 388 songs
with meter changes, 468 redundant signature events, and six changes off the
preceding bar grid. Musical distributions below use train only; the source-ID
integrity check across all splits found no shared MSD identities crossing splits.
This does not establish absence of near duplicates or lyric-decoder exposure.
Reproduce with `conda run --no-capture-output -n prosodia-lyricist python
outputs/lakh-octuple-design/profile.py`; results are in the adjacent `profile.json`.

Each note has the ordered tuple `(bar, position, pitch, interval, duration, gap,
meter, tempo)`. Metadata, targets, and masks are separate.

| Field | Definition | Encoding |
| --- | --- | --- |
| Bar | Global bar index minus the first retained note's bar index | 32-D sinusoidal encoding; no 256-bar table or wrapping |
| Position | Onset offset within its bar, in quarter-note units | 96 subdivisions/quarter; factor integer position into quarter index and subdivision, with 16-D embeddings each |
| Pitch | MIDI pitch, 0–127 | 32-D categorical embedding |
| Interval | Current pitch minus preceding melody pitch | Exact −127…127; `START` for the first note; 32-D embedding |
| Duration | `(offset_tick − onset_tick) / PPQ` | Fixed multiscale codebook; 32-D embedding |
| Gap | `(onset_tick − previous_offset_tick) / PPQ` | Same codebook, separate embedding; zero means connected notes, first note uses `START` |
| Meter | Explicit numerator and denominator at onset | Separate 16-D numerator and denominator embeddings; retain unreduced signature |
| Tempo | Quarter notes/minute at onset | Logarithmic codebook, 24 bins/octave from 8 through 512 QPM; 32-D embedding |

Use a fixed vocabulary, not a vocabulary learned from all splits. Reserve
distinct `PAD`, `UNK`, and `MASK` values; `START` is distinct from numeric zero.
Bar zero is a real bar, pitch interval zero a repeated pitch, and gap zero a real
absence of silence. A separate attention mask identifies padded notes.
Position supports quarter indices 0–1023 and subdivisions 0–95; meter supports
numerators 1–255 and denominators 1, 2, 4, 8, 16, 32, 64. This includes the
observed 149/4 metadata without inventing shorter bars. Flag unusual signatures
for review; representability does not validate their musical correctness.
Unsupported signatures fail strict preparation/inference. Missing meter uses
unknown bar/position/meter values only in an explicitly trained fallback mode.
Tempo has explicit underflow/overflow/unknown codes, never silent clipping.
Within range, the tempo index is `floor(24*log2(QPM/8) + 0.5)` (0–144).

Use raw ticks and PPQ, not seconds, as the rhythmic authority. The audited
notes use 960 PPQ; inference must also work at other PPQs. The tempo map stores
microseconds/quarter: `QPM = 60_000_000 / tempo`. Integrate that map for playback
seconds, including changes inside notes. Keep explicit/default/unknown tempo
provenance; MIDI's default tempo is 120 QPM, not an observed tempo event.

Use `Q = 96`. Ordinary nearest-grid quantization on the current training set
gives the following onset errors; the production position codec also handles
bar boundaries explicitly as described below.

| Subdivisions/quarter | Exact onsets | Mean absolute error (quarters) |
| --- | ---: | ---: |
| 16, original MusicBERT | 48.81% | 0.00830 |
| 48 | 60.61% | 0.00294 |
| 96, proposed | 74.41% | 0.00087 |

The higher resolution covers triplets and fine subdivisions without adding
sequence steps. These are encoding errors, not evidence of better prediction.
Duration/gap centers are `j / Q` for `j = 0…384`, followed by
`4 * 2**(k/48)` for `k = 1…384` (up to 1,024 quarters). Choose the nearest center,
ties upward; durations may only choose positive centers, while gaps may choose
zero. Above 1,024 use an explicit overflow code. The train maxima are 32.5
quarters for duration and 580.125 for gap. Never delete a very short note.

Include three source-only timing residuals alongside the tuple, projected by a
small MLP: position, duration, and gap. For true within-bar position `u`, bar
length `L = 4*n/d`, set `p = min(floor(Q*u + 0.5), Q*L - 1)` and residual
`Q*u - p`. This prevents rounding a pre-bar note into a downbeat. For a time
value `x` and selected center `c`, use `(x-c)/w`, with `w = 1/Q` for `c <= 4`
and `w = c*(2**(1/48)-1)` otherwise. Overflow uses `c = w = 1024`.
The first gap residual is zero. Residuals distinguish tiny positive gaps from
zero and preserve timing between centers. Ablate them separately. Keep the
unaltered ticks and maps for export; the tuple is not a lossless MIDI file format.

Build a single canonical bar grid for preparation and inference. Collapse
redundant same-signature events, reject conflicting signatures at one tick, and
use exact rational tick/PPQ arithmetic where possible. Tick zero is the origin
when a signature is available there. A genuine signature change closes any
partial bar and starts the next global bar; record off-bar changes. Keep empty
bars through instrumental gaps. Rebase only the integer bar index at the first
vocal note: preserve its within-bar position and never reset at lyric lines.
If the first explicit signature occurs later, its tick starts the known grid;
earlier notes have unknown meter and fail the initial strict configuration.
An explicit pickup/downbeat origin may override the default if supplied at both
training and inference; do not infer one from lyric text. A MIDI grid is not an
independent annotation of perceptual downbeats. The existing `meter_entries`
resets on every signature event, so regenerate meter features for both comparison
models using this common policy.

For example, semantic fields `(bar, position, pitch, interval, duration, gap,
meter, tempo)` for three notes can be `(0,0,60,START,1,START,4/4,120)`,
`(0,96,62,+2,0.5,0,4/4,120)`, `(0,192,64,+2,1,0.5,4/4,120)`.
Positions are grid units; duration/gap are shown in quarters before binning,
and tempo is shown before its log quantization. The last note follows a
half-quarter rest. No separate rest or bar tokens are inserted.

Concatenate the eight 32-D field vectors, project to 256 dimensions, add the
timing-residual projection and note-order positional encoding, then normalize.
Use the existing scale: four bidirectional Transformer layers, four heads,
FFN 1024, dropout 0.1, whole-song attention, maximum 2,048 notes. No second tower
or pooled contrastive vector. Train all melody parameters immediately. The
existing meter adapter is replaced by the meter/position fields. Small uniform
song transpositions in −5…+5 semitones, rejecting out-of-MIDI-range shifts, are
a separate augmentation experiment; the interval remains unchanged. Do not
change augmentation when measuring the encoding alone.

Keep the output contract: three lexical stress classes and IPA long/short
vowels using the existing historical pronunciation policy. A long note is not
automatically an IPA long vowel; a downbeat is not automatically lexical primary
stress. These targets express compatibility with observed lyrics, not a unique
correct template for any melody. Word-local strict alignment still masks
count-mismatched words for prosody CE while retaining every note as context.
Never feed words, phonemes, syllable IDs, continuation labels, target counts,
or alignment indices into the source encoder.

First compare the new encoding with `177-D + meter`, holding the Lakh split,
canonical meter policy, supplied phrase boundaries, heads, training settings,
and scoring anchors fixed. Then add a parallel `START / CONTINUE` head using
the raw continuation annotations. Train this head on all valid note roles,
including notes in words masked for IPA count mismatch; prosody CE remains
limited to aligned starts. Begin with `L = CE_stress + CE_length + 0.5*CE_role`,
each term normalized over its own valid labels. Role class weights may be
`N/(2*N_class)`, computed on train only; calibrate the continuation threshold on
validation and report unweighted CE too. This is a starting recipe, not a tuned
optimum. At inference, start a group at predicted starts and use its first note's
stress/length predictions for one syllable slot. Force each supplied phrase's
first note to start a group. Adjacent equal prosody pairs stay separate when
both are starts; there is no CTC-style collapse. This corrects one pair per note
without reading the reference syllable count. It still cannot emit multiple
syllables on one note; report that limitation and alignment coverage.

The initial pipeline is explicitly **phrase-conditioned**: the existing input
requires phrase-end markers, so known line membership can remain a source
feature in both comparison models. Annotated Lakh lyric lines are not magically
available in arbitrary MIDI. For unmarked inference, train a separate version
with no gold line embeddings and a phrase-start head supervised by line labels.
Predicted phrase starts must also be syllable starts. A rest threshold alone is
insufficient: 49.80% of training line boundaries have less than one quarter of
silence, and 7,959 have no silence. Keep supplied and predicted segmentation
results separate; reject over-limit output with a recorded reason rather than
silently forcing gold counts or truncating phrases. XAI keeps its 40-slot phrase
limit and its independent whole-song source/target limits.

The implementation follows these four steps:

1. Add a shared tick/meter adapter and versioned octuple codec under
   `melody_encoder/`. Extend `lakh.py`/`midi.py` to provide the same canonical
   notes/maps; keep processed vocal notes and source indices intact. Store role
   targets separately. Select a monophonic vocal track explicitly for new MIDI;
   arbitrary accompaniment tracks are outside this input contract.
2. Add typed tuple/residual batches in `bridge_data.py`, field embedding support
   in `melody_encoder/modeling.py`, and a distinct representation/configuration
   path in `bridge_model.py`. Use a new `bridge_lakh_octuple.yaml` and output
   directory. Version checkpoints and vocab/quantizer/grid-policy hashes;
   reject 177-D or masked-177-D weight imports. Preserve old experiment loading.
3. Preserve existing song-group split membership, audit canonical melody hashes,
   old content identities, normalized lyrics, and MSD aliases before freezing
   the new manifest. New collisions must be resolved as whole groups before any
   training. Update `split_integrity.py`: it currently fingerprints the 177-D
   encoder's inputs. Audit decoder training exposure separately for full-pipeline
   generalization claims. No new random row/phrase split and no DALI training data.
   Keep the phonetic-only export out of this first supervised experiment: it
   lacks the spelling alignment used to produce the shared historical IPA labels.
4. Extend `bridge_infer.py` and evaluation after the representation comparison:
   add role decoding, then unmarked segmentation if needed. Feed the resulting
   discrete template into the unchanged XAI decoder. Reference lyrics enter
   scoring only; template predictions are not teacher-forced from references.

Verify PPQ invariance; triplets and tiny notes; rest versus duration; meter
changes/redundancy; pickup origins; 6/8 versus 3/4; bars beyond 255; exact
preparation/MIDI tensor parity; padding; and invariance of encoder tensors when
target text/roles change under a fixed supplied-phrase contract. Test a
multi-note melisma and adjacent identical syllable templates explicitly.

Report three distinct evaluations: (1) aligned-anchor stress/length macro-F1,
CE, joint accuracy and coverage; (2) predicted-start count MAE, continuation
precision/recall/F1 and complete-template errors without gold counts; (3)
generated-lyric agreement with its supplied template, reference prosody, and
human singability. Always-start already attains 97.66% note-role accuracy on
the profiled train set, so raw role accuracy is inadequate. Include common-set
coverage, failures, tempo/meter strata, several matched seeds, and an oracle
template decoder control. Select models/thresholds on validation, then evaluate
test once. Test the 96-versus-48 grid, timing residuals, pitch interval, gap,
and tempo in staged ablations after the principal comparison.
