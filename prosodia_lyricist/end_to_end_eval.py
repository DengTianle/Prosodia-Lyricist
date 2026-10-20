"""Evaluate lyric generation, soft/hard conditioning, and matched melody controls."""

import argparse
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .bridge_baselines import evaluate_duration_baselines
from .bridge_data import decode_bridge_tokens, encode_bridge_targets
from .data import read_manifest
from .end_to_end_data import EndToEndCollator, EndToEndDataset
from .end_to_end_model import EndToEndProsody, condition_inputs
from .end_to_end_train import autocast, run_epoch, validate_precision
from .evaluation import evaluate_prosody
from .melody_checkpoint import file_sha256
from .runtime import seed_everything, select_device


@torch.inference_mode()
def evaluate_generation(
    model,
    examples,
    tokenizer,
    device,
    *,
    max_new_tokens,
    template_mode="soft",
    precision="fp32",
    progress=False,
):
    """Score actual generated lyrics with IPA, never the decoder's auxiliary predictions."""
    model.eval()
    collate = EndToEndCollator(tokenizer.pad_token_id)
    slots, phrases, count_matches, count_error = 0, 0, 0, 0
    correct = dict.fromkeys(("stress", "length", "pair"), 0)
    completed, failed, ipa_failures, source_slots, source_length_correct = 0, 0, 0, 0, 0
    equal_slots, equal_correct, equal_phrases = 0, 0, 0
    samples = []
    for example in tqdm(examples, desc=f"Generate {template_mode}", disable=not progress):
        inputs = {k: v.to(device) for k, v in collate([example]).items()}
        error, explanation = None, None
        try:
            with autocast(precision):
                generated = model.generate(
                    **condition_inputs(inputs),
                    tokenizer=tokenizer,
                    template_mode=template_mode,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            explanation = generated.explanations[0]
            lyrics = explanation["lines"]
            completed += int(explanation["completed"])
        except ValueError as exc:
            # Pronunciation failures during corrected decoding must not remove a song
            # from the denominator. Configuration/model errors are validated beforehand.
            if "IPA" not in str(exc) and "pronunc" not in str(exc).lower():
                raise
            error, lyrics = str(exc), []
            failed += 1
        metrics = evaluate_prosody(example["reference_lines"], lyrics, **(
            {"prosody_mode": "actual"} if model.prosody_mode == "actual" else {}
        ))
        ipa_failures += metrics["pronunciation_failures"]
        for index, phrase in enumerate(metrics["phrases"]):
            reference, actual = phrase["input_syllables"], phrase["generated_syllables"] or []
            phrases += 1
            count_matches += int(bool(reference) and len(reference) == len(actual))
            count_error += abs(len(reference) - len(actual))
            compared = max(len(reference), len(actual))
            slots += compared
            equal = index < example["melody"]["line_count"] and len(reference) == example["melody"][
                "note_line_ids"
            ].count(index + 1)
            if equal:
                equal_phrases += 1
                equal_slots += compared
            for truth, predicted in zip_longest(reference, actual):
                if truth is not None and predicted is not None:
                    s, n = (truth[key] == predicted[key] for key in ("stress", "length"))
                    correct["stress"] += s
                    correct["length"] += n
                    correct["pair"] += s and n
                    if equal:
                        equal_correct += n
        if explanation:
            preview = decode_bridge_tokens(
                explanation["melody_conditioning"]["hard_template_tokens"], pairs=model.bridge.pairs
            )
            # Reuse the same parsed actual lyrics when comparing against the input preview.
            actual_lines = [p["generated_syllables"] or [] for p in metrics["phrases"]]
            for source, actual in zip_longest(preview, actual_lines, fillvalue=None):
                source = source["syllables"] if source is not None else []
                actual = actual or []
                source_slots += max(len(source), len(actual))
                source_length_correct += sum(
                    a["length"] == b["length"] for a, b in zip(source, actual)
                )
        else:
            source_slots += sum(example["syllable_counts"])
        samples.append(
            dict(
                song_id=example["song_id"],
                lyrics=lyrics,
                error=error,
                completed=bool(explanation and explanation["completed"]),
                metrics=metrics,
                conditioning=explanation["melody_conditioning"] if explanation else None,
            )
        )
    return dict(
        songs=len(examples),
        phrases=phrases,
        compared_slots=slots,
        accuracy={name: n / slots if slots else None for name, n in correct.items()},
        equal_note_ipa_count=dict(
            phrases=equal_phrases,
            compared_slots=equal_slots,
            length_accuracy=equal_correct / equal_slots if equal_slots else None,
        ),
        phrase_count_accuracy=count_matches / phrases if phrases else None,
        phrase_count_mae=count_error / phrases if phrases else None,
        completed_songs=completed,
        failed_generation_songs=failed,
        pronunciation_failures=ipa_failures,
        hard_source_preview_length_agreement=(
            source_length_correct / source_slots if source_slots else None
        ),
        samples=samples,
    )


def melody_control_examples(examples):
    """Cyclic donors with the same phrase count; prompts/targets/counts stay fixed."""
    groups = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example["melody"]["line_count"]].append(index)
    original, changed, pairs = [], [], []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for index, donor in zip(indices, indices[1:] + indices[:1], strict=True):
            original.append(examples[index])
            changed.append({**examples[index], "melody": examples[donor]["melody"]})
            pairs.append(
                dict(song_id=examples[index]["song_id"], donor_song_id=examples[donor]["song_id"])
            )
    return original, changed, pairs


def evaluate_end_to_end(
    checkpoint,
    prepared_dir,
    *,
    split="test",
    device="auto",
    precision="fp32",
    batch_size=4,
    limit=None,
    generation_limit=None,
    count_source="ipa",
    max_new_tokens=None,
    compare_hard=False,
    melody_controls=False,
    progress=False,
    seed=1234,
):
    if (
        split not in ("valid", "test")
        or batch_size < 1
        or (limit is not None and limit < 1)
        or (generation_limit is not None and generation_limit < 1)
    ):
        raise ValueError("Use valid/test with positive batch size and optional positive limits")
    seed_everything(seed)
    device = select_device(device)
    validate_precision(device, precision)
    checkpoint, prepared_dir = Path(checkpoint), Path(prepared_dir)
    model = EndToEndProsody.load(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    path = checkpoint / "run.json"
    run = json.loads(path.read_text()) if path.exists() else {}
    cfg = run.get("config", {}).get("model", {})
    position_limit = model.template_model.bart.config.max_position_embeddings
    target_limit = cfg.get("max_target_length", position_limit)
    max_new_tokens = target_limit - 1 if max_new_tokens is None else max_new_tokens
    if not 1 <= max_new_tokens < target_limit:
        raise ValueError("max_new_tokens must be positive and below the checkpoint target limit")
    dataset = EndToEndDataset(
        prepared_dir,
        split,
        tokenizer,
        **model.dataset_options,
        max_source_length=cfg.get("max_source_length", position_limit),
        max_target_length=target_limit,
        count_source=count_source,
        limit=limit,
    )
    collate = EndToEndCollator(tokenizer.pad_token_id)

    def teacher(examples, mode="soft"):
        loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collate)
        return run_epoch(model, loader, device, precision=precision, template_mode=mode)

    def generate(examples, mode="soft"):
        return evaluate_generation(
            model,
            examples,
            tokenizer,
            device,
            precision=precision,
            template_mode=mode,
            max_new_tokens=max_new_tokens,
            progress=progress,
        )

    examples = dataset.examples[:generation_limit]
    manifest = read_manifest(prepared_dir)
    report = dict(
        schema_version=1,
        conditioning="parallel_soft_template",
        decoder_mode=model.decoder_mode,
        prosody_mode=model.prosody_mode,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256={
            "template": file_sha256(checkpoint / "weights.pt"),
            "bridge": file_sha256(checkpoint / "bridge" / "bridge_weights.pt"),
        },
        provenance=model.provenance,
        pretraining_split_check=run.get("pretraining_split_check"),
        template_split_verification=run.get("template_split_verification", "unverified"),
        data=dict(
            prepared_dir=str(prepared_dir.resolve()),
            split=split,
            split_sha256=manifest["sha256"][f"{split}.jsonl"],
            count_source=count_source,
            songs=len(dataset),
            generation_songs=len(examples),
            skipped_songs=dataset.skipped,
            skipped_limits=dataset.skipped_limits,
        ),
        runtime=dict(
            device=str(device),
            precision=precision,
            batch_size=batch_size,
            seed=seed,
            max_new_tokens=max_new_tokens,
        ),
        teacher_forced=teacher(dataset),
        generation=generate(examples),
        metric_definitions={
            "generation_accuracy": "Actual generated IPA versus reference IPA, by phrase/slot. "
            "Missing/extra slots and failed songs count as incorrect. This is reference agreement, "
            "not a measure of all acceptable lyrics or physical singability.",
            "teacher_forced": (
                "Reference word history enters only the lyric decoder. "
                if model.decoder_mode == "xai"
                else "Reference word/prosody history enters only the lyric decoder. "
            ) + "The parallel bridge never receives reference prosody labels as input.",
            "bridge_accuracy": "Hard argmax of parallel slots, with no label history. "
            "With note counts, only equal note/IPA count phrases have aligned reference labels.",
            "source_agreement": "Diagnostic agreement with the argmax input preview; "
            "it is not ground-truth accuracy.",
            "duration_baselines": "Same songs as lyric generation, scored on reference slots. "
            "Generated lyrics additionally incur penalties for extra output slots.",
        },
    )
    report["duration_baselines"] = evaluate_duration_baselines(
        [
            {
                **e["melody"],
                "labels": encode_bridge_targets(
                    e["reference_lines"], model.bridge.max_syllables, pairs=model.bridge.pairs
                ),
                "note_duration_seconds": np.asarray(e["note_duration_seconds"], dtype=np.float64),
                "song_mean_note_duration_seconds": float(np.mean(e["note_duration_seconds"])),
            }
            for e in examples
        ],
        lines_per_window=model.bridge.encoder_lines_per_window,
        pairs=model.bridge.pairs,
    )
    if compare_hard:
        report["hard_template_ablation"] = dict(
            teacher_forced=teacher(dataset, "hard"),
            generation=generate(examples, "hard"),
        )
    if melody_controls:
        actual, shuffled, pairs = melody_control_examples(dataset.examples)
        controls = dict(
            paired_songs=len(actual), unpaired_songs=len(dataset) - len(actual), pairs=pairs
        )
        if actual:
            controls.update(
                correct_melody=teacher(actual),
                wrong_melody=teacher(shuffled),
                correct_generation=generate(actual[:generation_limit]),
                wrong_generation=generate(shuffled[:generation_limit]),
            )
        report["melody_controls"] = controls
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Maximum retained songs for all metrics")
    parser.add_argument(
        "--generation-limit", type=int, help="Optional subset for slow lyric generation"
    )
    parser.add_argument("--count-source", choices=("ipa", "notes"), default="ipa")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--compare-hard", action="store_true")
    parser.add_argument("--melody-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True, type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    if output.exists():
        parser.error("Choose a new output path")
    report = evaluate_end_to_end(**args, progress=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"Lyric generation accuracy: {report['generation']['accuracy']}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
