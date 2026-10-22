"""Class diagnostics distinguish majority collapse from actual minority learning."""

import math

import pytest
import torch

from prosodia_lyricist.bridge_metrics import ClassificationMetrics


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    )),
])
def test_constant_predictions_report_class_counts_recall_and_probabilities(device):
    metrics = ClassificationMetrics(("strong", "substrong", "weak"), collect_length_scores=True)
    # Reference pairs: strong-long, substrong-short, weak-short, weak-short.
    expected = torch.tensor([0, 3, 5, 5], device=device)
    predicted = torch.full_like(expected, 5)
    strength = torch.tensor([[0.1, 0.2, 0.7]], device=device).log().expand(4, -1)
    length = torch.tensor([[0.3, 0.7]], device=device).log().expand(4, -1)
    for start, end in ((0, 1), (1, 4)):
        metrics.update(
            predicted[start:end],
            expected[start:end],
            strength_logits=strength[start:end],
            length_logits=length[start:end],
        )
    report = metrics.report()
    stress = report["strength"]
    assert stress["slots"] == 4
    assert stress["confusion_matrix"] == [[0, 0, 1], [0, 0, 1], [0, 0, 2]]
    assert stress["balanced_accuracy"] == pytest.approx(1 / 3)
    assert stress["macro_f1"] == pytest.approx((2 / 3) / 3)
    assert stress["classes"]["substrong"]["recall"] == 0
    assert stress["classes"]["weak"]["mean_probability"] == pytest.approx(0.7)
    length = report["length"]
    assert length["confusion_matrix"] == [[0, 1], [0, 3]]
    assert length["majority_class"] == "short"
    assert length["majority_accuracy"] == 0.75
    assert length["balanced_accuracy"] == 0.5
    assert length["classes"]["long"]["predicted_fraction"] == 0
    assert length["classes"]["long"]["recall"] == 0
    assert length["classes"]["long"]["mean_probability"] == pytest.approx(0.3)
    assert length["label_entropy"] == pytest.approx(-0.25 * math.log(0.25) - 0.75 * math.log(0.75))
    assert length["ranking"]["roc_auc"] == 0.5
    assert length["ranking"]["average_precision"] == 0.25


def test_all_short_predictions_can_still_rank_lengths_perfectly():
    metrics = ClassificationMetrics(("strong", "weak"), collect_length_scores=True)
    # All probabilities of long are below the argmax boundary, but long ranks first.
    probabilities = torch.tensor([[0.4, 0.6], [0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])
    metrics.update(
        torch.ones(4, dtype=torch.long),
        torch.tensor([0, 1, 1, 1]),
        length_logits=probabilities.log(),
    )
    length = metrics.report()["length"]
    assert length["classes"]["short"]["predicted_fraction"] == 1
    assert length["classes"]["long"]["recall"] == 0
    assert length["ranking"]["roc_auc"] == 1
    assert length["ranking"]["average_precision"] == 1


def test_absent_reference_classes_do_not_reduce_balanced_accuracy():
    metrics = ClassificationMetrics(("strong", "substrong", "weak"))
    expected = torch.tensor([1, 1])
    metrics.update(expected, expected)
    result = metrics.report()
    assert result["strength"]["balanced_accuracy"] == 1
    assert result["strength"]["classes"]["substrong"]["recall"] is None
    assert result["strength"]["classes"]["substrong"]["f1"] is None
    assert result["length"]["macro_f1"] == 1
    assert result["length"]["label_entropy"] == 0
    assert result["length"]["classes"]["short"]["mean_probability"] is None


def test_empty_diagnostics_are_serializable_and_have_no_fake_accuracy():
    import json

    result = ClassificationMetrics(("strong", "weak")).report()
    assert result["length"]["slots"] == 0
    assert result["length"]["majority_accuracy"] is None
    assert result["length"]["balanced_accuracy"] is None
    json.dumps(result, allow_nan=False)
