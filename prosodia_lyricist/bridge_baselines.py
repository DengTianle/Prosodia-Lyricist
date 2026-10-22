"""Duration-only length baselines on the bridge's reference IPA skeleton."""

from collections import Counter

import numpy as np

from .bridge_data import PAIRS, decode_bridge_tokens

SCOPES = ("global", "song", "window", "line")


def evaluate_duration_baselines(examples, *, lines_per_window, generation=None, pairs=PAIRS):
    """Compare raw note duration to arithmetic means, without inspecting target labels.

    Like bridge generation, the comparison uses known IPA counts. Notes and slots
    are paired in phrase order; extra notes are trimmed and absent notes are wrong.
    The equal-count subset also reports the model's accuracy on those same phrases.
    """
    if not examples or lines_per_window < 1:
        raise ValueError("Duration baselines require examples and positive window size")
    global_mean = float(
        sum(np.sum(e["note_duration_seconds"]) for e in examples)
        / sum(len(e["note_duration_seconds"]) for e in examples)
    )
    totals = {name: Counter() for name in ("all_ipa_slots", "equal_note_ipa_count")}
    missing, extra, masked = 0, 0, 0
    aligned = "target_note_indices" in examples[0]
    for example in examples:
        durations = np.asarray(example["note_duration_seconds"], dtype=np.float64)
        line_ids = np.asarray(example["note_line_ids"]) - 1
        window_ids = line_ids // lines_per_window
        line_means = np.bincount(line_ids, weights=durations) / np.bincount(line_ids)
        window_means = np.bincount(window_ids, weights=durations) / np.bincount(window_ids)
        thresholds = {
            "global": global_mean,
            "song": example["song_mean_note_duration_seconds"],
            "window": window_means[window_ids],
            "line": line_means[line_ids],
        }
        predictions = {scope: durations > threshold for scope, threshold in thresholds.items()}
        offset = 0
        for index, line in enumerate(decode_bridge_tokens(example["labels"], pairs=pairs)):
            truth = np.asarray([s["length"] == "long" for s in line["syllables"]])
            reference_count = len(truth)
            note_mask = line_ids == index
            note_count = int(note_mask.sum())
            if aligned:
                anchors = np.asarray(example["target_note_indices"][offset + 1:
                                    offset + 1 + reference_count])
                active = anchors >= 0
                offset += reference_count + 1
                masked += int((~active).sum())
                truth = truth[active]
                extra += note_count - len(truth)
                if not len(truth):
                    continue
                correct = {
                    scope: int(np.sum(predicted[anchors[active]] == truth))
                    for scope, predicted in predictions.items()
                }
            else:
                overlap = min(note_count, len(truth))
                missing += max(0, len(truth) - note_count)
                extra += max(0, note_count - len(truth))
                correct = {
                    scope: int(np.sum(predicted[note_mask][:overlap] == truth[:overlap]))
                    for scope, predicted in predictions.items()
                }
            correct.update(always_long=int(truth.sum()), always_short=int((~truth).sum()))
            subsets = ["all_ipa_slots"]
            if note_count == reference_count:
                subsets.append("equal_note_ipa_count")
            for subset in subsets:
                totals[subset].update(phrases=1, compared_slots=len(truth), **correct)

    scores = {}
    for subset, counts in totals.items():
        model = generation if subset == "all_ipa_slots" else (
            generation.get(subset) if generation else None
        )
        model_accuracy = None
        if model is not None:
            if any(model[key] != counts[key] for key in ("phrases", "compared_slots")):
                raise ValueError("Duration baseline coverage differs from model generation")
            # Accept both evaluate_templates results and serialized bridge_eval reports.
            model_accuracy = model.get("length_accuracy", model.get("accuracy", {}).get("length"))
        slots = counts["compared_slots"]
        scores[subset] = {
            "phrases": counts["phrases"],
            "compared_slots": slots,
            "model_generation_length_accuracy": model_accuracy,
            "baselines": {},
        }
        for scope in (*SCOPES, "always_long", "always_short"):
            accuracy = counts[scope] / slots if slots else None
            scores[subset]["baselines"][scope] = {
                "correct_slots": counts[scope],
                "length_accuracy": accuracy,
                "minus_model_length_accuracy": (
                    accuracy - model_accuracy
                    if accuracy is not None and model_accuracy is not None else None
                ),
            }
    return {
        "prediction_rule": "long iff raw note duration > arithmetic mean; ties are short",
        "threshold_scopes": {
            "global": "All notes in retained evaluation examples; no target labels used.",
            "song": "All notes in the prepared song, including other legacy encoder windows.",
            "window": f"Consecutive {lines_per_window}-line encoder windows, including short tail.",
            "line": "All notes in the current phrase.",
        },
        "global_mean_note_duration_seconds": global_mean,
        "slot_count_source": "ipa",
        "alignment": (
            "Use the same observed first-note-per-sung-group anchors and word-level target "
            "mask as the note heads; unmatched IPA slots are excluded from both scores."
            if aligned else
            "Pair notes and IPA slots by phrase/slot order using reference IPA counts. "
            "Trim extra notes; missing-note slots count as incorrect. Equal counts do not "
            "guarantee phonetic alignment; compare the model and baseline on the same subset."
        ),
        "missing_note_slots": missing,
        "extra_notes_ignored": extra,
        "masked_ipa_slots": masked,
        **scores,
    }
