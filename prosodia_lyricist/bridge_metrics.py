"""Template metrics shared by parallel forward evaluation and legacy generation."""

from collections import Counter
from itertools import zip_longest

from .bridge_data import decode_bridge_tokens
from .evaluation import prosody_bleu


class TemplateMetrics:
    def __init__(self, *, pairs, bridge_scope):
        self.pairs, self.bridge_scope = pairs, bridge_scope
        self.counts = Counter()

    def diagnose_counts(self, counts, note_counts):
        active = counts.gt(0)
        self.counts.update(
            count_matches=int((note_counts.eq(counts) & active).sum()),
            count_error=int((note_counts - counts).abs()[active].sum()),
            count_phrases=int(active.sum()),
        )

    def record(self, sequences, labels, note_counts):
        c = self.counts
        for predicted, truth, phrase_notes in zip(sequences, labels, note_counts, strict=True):
            expected = decode_bridge_tokens([t for t in truth if t != -100], pairs=self.pairs)
            actual = decode_bridge_tokens(predicted, pairs=self.pairs)
            for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
                hypothesis, reference = a["syllables"], b["syllables"]
                c.update(
                    phrases=1,
                    exact=int(a == b),
                    bleu=prosody_bleu(reference, hypothesis),
                    short_phrases=int(min(len(reference), len(hypothesis)) < 4),
                    slots=max(len(hypothesis), len(reference)),
                )
                equal_count = phrase_notes[index] == len(reference)
                if equal_count:
                    c.update(
                        equal_count_phrases=1,
                        equal_count_slots=max(len(hypothesis), len(reference)),
                    )
                for predicted_slot, expected_slot in zip_longest(hypothesis, reference):
                    if predicted_slot is not None and expected_slot is not None:
                        length_correct = predicted_slot["length"] == expected_slot["length"]
                        c.update(
                            strength_correct=int(
                                predicted_slot["stress"] == expected_slot["stress"]
                            ),
                            length_correct=int(length_correct),
                            pair_correct=int(predicted_slot == expected_slot),
                            equal_count_length_correct=int(equal_count and length_correct),
                        )

    def report(self):
        c = self.counts
        return {
            "phrases": c["phrases"],
            "slot_count_source": "ipa",
            **{
                f"{name}_accuracy": c[f"{name}_correct"] / c["slots"] if c["slots"] else None
                for name in ("strength", "length", "pair")
            },
            "compared_slots": c["slots"],
            "equal_note_ipa_count": {
                "phrases": c["equal_count_phrases"],
                "compared_slots": c["equal_count_slots"],
                "length_accuracy": (
                    c["equal_count_length_correct"] / c["equal_count_slots"]
                    if c["equal_count_slots"]
                    else None
                ),
            },
            "exact_template_accuracy": c["exact"] / c["phrases"] if c["phrases"] else None,
            "prosody_bleu": c["bleu"] / c["phrases"] if c["phrases"] else None,
            "bleu_short_phrases": c["short_phrases"],
            "count_diagnostic_phrases": c["count_phrases"],
            "note_ipa_count_match_rate": (
                c["count_matches"] / c["count_phrases"] if c["count_phrases"] else None
            ),
            "note_ipa_count_mae": (
                c["count_error"] / c["count_phrases"] if c["count_phrases"] else None
            ),
            f"skipped_slot_limit_{'songs' if self.bridge_scope == 'song' else 'windows'}": (
                c["skipped_examples"]
            ),
            "skipped_slot_limit_phrases": c["skipped_phrases"],
            "skipped_target_limit_examples": c["skipped_target_examples"],
            "skipped_target_limit_phrases": c["skipped_target_phrases"],
            "example_scope": self.bridge_scope,
        }
