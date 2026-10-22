"""Template metrics shared by parallel forward evaluation and legacy generation."""

from collections import Counter
from itertools import zip_longest
from math import log

import torch

from .bridge_data import decode_bridge_tokens
from .evaluation import prosody_bleu


class ClassificationMetrics:
    """Per-head class balance, confusion, and calibration diagnostics at valid slots."""

    def __init__(self, strength_labels, *, collect_length_scores=False):
        self.labels = {"strength": tuple(strength_labels), "length": ("long", "short")}
        self.confusion = {
            name: torch.zeros(len(labels), len(labels), dtype=torch.long)
            for name, labels in self.labels.items()
        }
        self.probability_sums = {
            name: torch.zeros(len(labels), dtype=torch.float64)
            for name, labels in self.labels.items()
        }
        self.probability_counts = Counter()
        self.collect_length_scores = collect_length_scores
        self.length_scores, self.length_targets = [], []

    @torch.no_grad()
    def update(self, predicted, expected, *, strength_logits=None, length_logits=None):
        for name, predictions, targets, logits in (
            ("strength", predicted // 2, expected // 2, strength_logits),
            ("length", predicted % 2, expected % 2, length_logits),
        ):
            size = len(self.labels[name])
            self.confusion[name] += (
                torch.bincount(targets * size + predictions, minlength=size * size)
                .reshape(size, size)
                .cpu()
            )
            if logits is not None:
                probabilities = logits.float().softmax(-1)
                # MPS cannot cast to float64; keep double-precision accumulation on CPU.
                self.probability_sums[name] += probabilities.sum(0).cpu().double()
                self.probability_counts[name] += len(targets)
                if name == "length" and self.collect_length_scores:
                    self.length_scores.append(probabilities[:, 0].cpu())
                    self.length_targets.append(targets.eq(0).cpu())

    def length_ranking(self):
        result = {"positive_class": "long", "roc_auc": None, "average_precision": None}
        if not self.length_scores:
            return result
        scores = torch.cat(self.length_scores)
        targets = torch.cat(self.length_targets)
        positives = int(targets.sum())
        negatives = len(targets) - positives
        if not positives:
            return result
        order = scores.argsort(descending=True)
        sorted_scores = scores[order]
        # Group ties so constant scores give ROC AUC 0.5 and AP equal to prevalence.
        ends = torch.cat((sorted_scores[:-1].ne(sorted_scores[1:]), torch.tensor([True])))
        indices = ends.nonzero(as_tuple=True)[0]
        true_positives = targets[order].double().cumsum(0)[indices]
        false_positives = indices.double() + 1 - true_positives
        preceding = torch.cat((torch.zeros(1, dtype=torch.float64), true_positives[:-1]))
        result["average_precision"] = float(
            (
                (true_positives - preceding)
                / positives
                * true_positives
                / (true_positives + false_positives)
            ).sum()
        )
        if negatives:
            zero = torch.zeros(1, dtype=torch.float64)
            result["roc_auc"] = float(
                torch.trapezoid(
                    torch.cat((zero, true_positives / positives)),
                    torch.cat((zero, false_positives / negatives)),
                )
            )
        return result

    def report(self):
        report = {}
        for name, labels in self.labels.items():
            confusion = self.confusion[name]
            references, predictions = confusion.sum(1), confusion.sum(0)
            total = int(references.sum())
            classes = {}
            for index, label in enumerate(labels):
                true, predicted = int(references[index]), int(predictions[index])
                correct = int(confusion[index, index])
                classes[label] = {
                    "reference_count": true,
                    "predicted_count": predicted,
                    "reference_fraction": true / total if total else None,
                    "predicted_fraction": predicted / total if total else None,
                    "precision": correct / predicted if predicted else 0.0,
                    "recall": correct / true if true else None,
                    "f1": 2 * correct / (true + predicted) if true + predicted else None,
                    "mean_probability": (
                        float(self.probability_sums[name][index]) / self.probability_counts[name]
                        if self.probability_counts[name]
                        else None
                    ),
                }
            supported = [value for value in classes.values() if value["reference_count"]]
            report[name] = {
                "labels": list(labels),
                "confusion_matrix": confusion.tolist(),
                "confusion_axes": "reference_rows_prediction_columns",
                "slots": total,
                "classes": classes,
                "majority_class": labels[int(references.argmax())] if total else None,
                "majority_accuracy": int(references.max()) / total if total else None,
                "label_entropy": (
                    -sum(
                        (int(count) / total) * log(int(count) / total)
                        for count in references
                        if count
                    )
                    if total
                    else None
                ),
                "balanced_accuracy": (
                    sum(value["recall"] for value in supported) / len(supported)
                    if supported
                    else None
                ),
                "macro_f1": (
                    sum(value["f1"] for value in supported) / len(supported) if supported else None
                ),
                "averaging": "classes_with_reference_support",
            }
        if self.collect_length_scores:
            report["length"]["ranking"] = self.length_ranking()
        return report


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

    def record(self, sequences, labels, note_counts, target_mask=None):
        c = self.counts
        for row, (predicted, truth, phrase_notes) in enumerate(
            zip(sequences, labels, note_counts, strict=True)
        ):
            expected = decode_bridge_tokens([t for t in truth if t != -100], pairs=self.pairs)
            actual = decode_bridge_tokens(predicted, pairs=self.pairs)
            offset = 0
            for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
                hypothesis, reference = a["syllables"], b["syllables"]
                mask = (
                    target_mask[row][offset + 1:offset + 1 + len(reference)]
                    if target_mask is not None else [True] * max(len(hypothesis), len(reference))
                )
                offset += len(reference) + 1
                if target_mask is not None and len(hypothesis) != len(reference):
                    raise ValueError("Aligned template metrics require the reference IPA skeleton")
                c["masked_slots"] += len(mask) - sum(mask)
                if not any(mask):
                    c["unsupervised_phrases"] += 1
                    continue
                complete = all(mask)
                c.update(
                    phrases=1,
                    template_phrases=int(complete),
                    exact=int(complete and a == b),
                    bleu=prosody_bleu(reference, hypothesis) if complete else 0,
                    short_phrases=int(complete and min(len(reference), len(hypothesis)) < 4),
                    slots=sum(mask),
                )
                equal_count = phrase_notes[index] == len(reference)
                if equal_count:
                    c.update(
                        equal_count_phrases=1,
                        equal_count_slots=sum(mask),
                    )
                for position, (predicted_slot, expected_slot) in enumerate(
                    zip_longest(hypothesis, reference)
                ):
                    if not mask[position]:
                        continue
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
            "exact_template_accuracy": (
                c["exact"] / c["template_phrases"] if c["template_phrases"] else None
            ),
            "prosody_bleu": c["bleu"] / c["template_phrases"] if c["template_phrases"] else None,
            "fully_supervised_phrases": c["template_phrases"],
            "unsupervised_phrases": c["unsupervised_phrases"],
            "masked_slots": c["masked_slots"],
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
