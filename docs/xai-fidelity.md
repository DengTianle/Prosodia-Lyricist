# Explainable Prosody implementation notes

The specification for the default model is the paper's Sections 3.1, 3.3,
3.4, and Eq. (1), together with the supplement's IPA rules. The adjacent
`../XAI-Lyricist` checkout is a partial implementation, not the final-model
oracle. The PDFs are in `docs/references/`.

## Implemented paper behavior

| Paper behavior | Implementation |
| --- | --- |
| Title, ordered sentences, keywords, and syllable prosody | Encoder sequence `<title>…<sent_0><keywords>…<prosody>…`; independent compound stress/length embeddings |
| Four output vocabularies | BART lyric vocabulary, word syllable count `0..max_syllables`, stress `pad/strong/weak`, length `pad/long/short` |
| Compound decoder input | Concatenate lyric, syllable-count, stress, and length embeddings; linear projection to hidden size |
| Four training targets | Sum of four independently averaged cross-entropies with unit default weights; no remainder/sentence loss |
| Non-word symbols | Prosody class 0 at BOS/EOS, punctuation, and intermediate BPE pieces; batch padding `-100` excluded from CE |
| Word prosody | Count IPA syllables; strong if any syllable is stressed; long if any syllable is long (Section 3.1) |
| IPA extraction | Primary and secondary stress map to strong; long-vowel marks and English diphthongs map to long |
| Sampling | Lyric symbol first; three non-pad prosody symbols at word completion; greedy or top-k temperature sampling |
| Prosody correction | Replace sampled labels with complete-word IPA labels before embedding the next decoder input |
| Explainable output | All four streams, predicted versus corrected labels, full IPA syllables, and source MIDI features available in JSON |

All target streams are shifted together. The current target cannot enter the
input used to predict itself. Inference caches only previously consumed events;
the corrected word-completion event is embedded on the next step, so the cache
never contains an uncorrected version of that event.

## BART word completion adaptation

The paper describes a word vocabulary and one event per word. Public BART uses
byte-pair subwords. Treating each piece as an independently pronounceable word
would corrupt IPA counts, especially for contractions and uncommon words.

This implementation encodes a word's BPE pieces, then an explicit `<word_end>`
event with its three labels. Earlier pieces have auxiliary pad labels. At that
event, the decoder samples all three labels, queries the complete word's IPA,
corrects disagreements, and uses the corrected event to generate the next word.
A whitespace-starting lexical token cannot start a second word before completion;
BOS/EOS and encoder metadata cannot occur inside a word. A 24-piece word limit
forces completion of otherwise unbounded fragments. Unknown pronunciations and
out-of-vocabulary counts fail explicitly when correction is enabled. A token
budget that cuts a word short removes the fragment and reports it in the result.

Consequences: this adds an event per word and changes token-budget usage and CE
normalization relative to a literal word-vocabulary model. Prosody heads predict
one word-level attribute set, while explanations additionally retain the complete
syllable-level IPA pattern. No hard melody-alignment constraint is claimed.

## Defaults and reproducibility limits

The default optimizer is Adam with betas (0.9, 0.98), epsilon 1e-5, no weight
decay, and warmup followed by constant LR. Embedding and Transformer dropout
are 0.3. The old reference `num_training_steps=-1` schedule, which stops learning
after warmup, is retained only as an explicit `xai_original` experiment.

This is an implementation of the explainable generation setup, not a numerical
reproduction of the published experiment:

- Training uses DALI rather than the paper's 101,120-song text corpus. Grouped
  splits, alignment validation, and whole-song rejection change corpus membership.
- The backbone is public pretrained `facebook/bart-base`; its layer/head/FFN
  dimensions are retained, rather than rebuilding the paper's stated architecture.
  Its tokenizer and pretrained weights differ from unavailable reference artifacts.
- DALI provides no keyword prompts. Their encoder slots remain empty.
- Sentence IDs support up to 256 lines; the configurable syllable vocabulary
  defaults to 40, while the paper reports 20. Source and target limits default
  to 1024 events and skip whole overlength songs.
- Modern Prosodic selects the first pronunciation. Its version and the binary
  stress/diphthong rule version are saved in preparation provenance.
- MIDI inference still assumes a monophonic melody with phrase-end markers and
  one note per syllable. Beat inference, melisma alignment, and the paper's
  musical-score visualization are not added by this model change.
- Training remains single-device full precision with early stopping; optimizer
  resume and distributed training are not implemented.

## Data and checkpoint migration

Prepared schema 4 records the revised IPA rules. Run preparation again and train
a new explainable checkpoint; old feature labels and plain-text decoder weights
cannot be silently reused as trained explanation heads. New checkpoints carry
an explicit format version, word-boundary token, IPA rules, all heads/embeddings,
and loss weights.

Old song-level Prosodia checkpoints still load through `legacy_model.py` and use
`legacy_features.py` at inference. `model.decoder_mode: lyrics` explicitly selects
that historical architecture for new experiments. Its old loss names are valid
only in that mode. Checkpoints from the adjacent research checkout have never
been compatible with this project's checkpoint format.

## Validation

The tests cover four-stream alignment and padding, the exact CE sum, gradients
through all heads and compound embeddings, causal shifting, strict save/reload,
actual cached decoding, predicted-versus-corrected feedback before the next word,
IPA failures, and a prepare/train/MIDI-infer integration run. Historical pipeline
regressions remain in the suite under explicit legacy mode. These checks validate
implementation behavior; they do not establish lyric quality or paper metrics.
