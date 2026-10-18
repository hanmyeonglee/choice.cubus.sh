"""Audit exact prepared input lengths and evaluate the production decision mapping."""

import argparse
import gc
import json
import platform
import resource
import time
from pathlib import Path
import math

import torch
from gliformer.gliformer import resolve_gliformer_collator_class
from pydantic import ValidationError
from torch.utils.data import DataLoader

from legacy_schemas import load_gliformer as load_model
from legacy_schemas import EvaluateRequest, choice_labels, state_text
from precision import evaluate


def audit(model, items, output):
    original_max_len = model.config.max_len
    # Count the complete prepared sequence, without the library's native truncation.
    model.config.max_len = 1_000_000
    model.data_processor.config.max_len = 1_000_000
    collator_cls = model.data_collator_class or resolve_gliformer_collator_class(model.config)
    collator = collator_cls(model.config, data_processor=model.data_processor, return_tokens=True, prepare_labels=False)
    with output.open("w") as target:
        for index, item in enumerate(items):
            body = item["request"]
            question = body["questions"]["q"]
            labels = choice_labels(question["criteria"]) if question["type"] == "choice" else [question["instructions"]]
            classes = {question["instructions"]: labels} if question["type"] == "choice" else labels
            words, _, _ = model.prepare_inputs([state_text(body["state"])])
            inputs = model._build_inference_input(words, classes=classes)
            batch = collator(inputs)
            length = int(batch["attention_mask"][0].sum().item())
            failures = []
            try:
                EvaluateRequest.model_validate(body)
            except ValidationError as error:
                failures = [{"loc": list(entry["loc"]), "type": entry["type"]} for entry in error.errors()]
            row = {"id": item["id"], "dataset": item["dataset"], "tokens": length,
                   "over_1024": length > 1024, "api_valid": not failures,
                   "api_errors": failures, "native_max_len": original_max_len,
                   "state_chars": len(state_text(body["state"]))}
            target.write(json.dumps(row) + "\n")
            if (index + 1) % 500 == 0:
                target.flush()
                print(json.dumps({"stage": "audit", "done": index + 1, "total": len(items)}), flush=True)
    model.config.max_len = original_max_len
    model.data_processor.config.max_len = original_max_len


@torch.inference_mode()
def batch_answers(model, items, batch_size):
    words, _, _ = model.prepare_inputs([state_text(item["request"]["state"]) for item in items])
    inputs = []
    for item, tokens in zip(items, words):
        question = item["request"]["questions"]["q"]
        labels = choice_labels(question["criteria"]) if question["type"] == "choice" else [question["instructions"]]
        classes = {question["instructions"]: labels} if question["type"] == "choice" else labels
        inputs.extend(model._build_inference_input([tokens], classes=classes))
    collator_cls = model.data_collator_class or resolve_gliformer_collator_class(model.config)
    collator = collator_cls(model.config, data_processor=model.data_processor, return_tokens=True, prepare_labels=False)
    loader = DataLoader(inputs, batch_size=batch_size, shuffle=False, collate_fn=collator)
    decoded, _ = model._process_multitask_batches(loader, threshold=-1.0, flat_ner=True, multi_label=True)
    predictions = model._collapse_single_group_results(decoded["classification"])
    answers = []
    for item, prediction in zip(items, predictions):
        question = item["request"]["questions"]["q"]
        scores = {entry["class_name"]: float(entry["score"]) for entry in prediction}
        if question["type"] == "noul":
            answer = {"type": "noul", "noul": scores[question["instructions"]]}
        else:
            labels = choice_labels(question["criteria"])
            total = sum(scores[label] for label in labels)
            if not math.isfinite(total) or total <= 0:
                raise ValueError("Invalid model score total.")
            probabilities = {name: scores[label] / total for name, label in zip(question["criteria"], labels)}
            answer = {"type": "choice", "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}
        answers.append(answer)
    if len(answers) != len(items):
        raise ValueError("The model returned the wrong number of batched answers.")
    return answers


def verify_batch(model, items, batch_size):
    examples = []
    seen = set()
    for item in items:
        if item["dataset"] not in seen:
            examples.append(item)
            seen.add(item["dataset"])
    single = [evaluate(model, item["request"])["q"] for item in examples]
    batched = batch_answers(model, examples, batch_size)
    largest_delta = 0
    for first, second in zip(single, batched):
        if first["type"] == "choice":
            if first["choice"] != second["choice"]:
                raise ValueError("Batched and single-request choices differ during verification.")
            differences = [abs(value - second["probabilities"][key]) for key, value in first["probabilities"].items()]
        else:
            if (first["noul"] >= 0.5) != (second["noul"] >= 0.5):
                raise ValueError("Batched and single-request noul decisions differ during verification.")
            differences = [abs(first["noul"] - second["noul"])]
        largest_delta = max(largest_delta, *differences)
    if largest_delta > 0.0001:
        raise ValueError(f"Batched verification score delta is too large: {largest_delta}")
    return {"verified_cases": len(examples), "max_score_delta": largest_delta}


def run(model, items, output, batch_size):
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            done.add(json.loads(line)["id"])
    pending = [item for item in items if item["id"] not in done]
    with output.open("a") as target:
        for index in range(0, len(pending), batch_size):
            chunk = pending[index:index + batch_size]
            started = time.perf_counter()
            try:
                answers = batch_answers(model, chunk, batch_size) if batch_size > 1 else [evaluate(model, chunk[0]["request"])["q"]]
                error = None
            except Exception as caught:
                answers = [None] * len(chunk)
                error = {"error_type": type(caught).__name__, "error": str(caught)}
                gc.collect()
            elapsed = time.perf_counter() - started
            for item, answer in zip(chunk, answers):
                write_prediction(target, item, answer, error, elapsed / len(chunk))
            target.flush()
            if (index + len(chunk)) % 100 == 0:
                print(json.dumps({"stage": "evaluate", "done": len(done) + index + len(chunk), "total": len(items), "batch_seconds": elapsed}), flush=True)


def write_prediction(target, item, answer, error, seconds):
    row = {"id": item["id"], "dataset": item["dataset"], "gold": item["gold"], "group": item["group"]}
    if answer is not None:
        if answer["type"] == "noul":
            prediction = answer["noul"] >= 0.5
            probabilities = {"true": answer["noul"], "false": 1 - answer["noul"]}
            gold_key = "true" if item["gold"] else "false"
        else:
            prediction = answer["choice"]
            probabilities = answer["probabilities"]
            gold_key = item["gold"]
        row.update(status="ok", answer=answer, prediction=prediction, correct=prediction == item["gold"],
                   confidence=max(probabilities.values()),
                   brier=sum((score - int(label == gold_key)) ** 2 for label, score in probabilities.items()))
    else:
        row.update(status="error", **error)
    row["seconds"] = seconds
    row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    target.write(json.dumps(row) + "\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "evaluate"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verify-batch", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in args.items.read_text().splitlines() if line.strip()]
    started = time.perf_counter()
    model = load_model(args.checkpoint, "cpu")
    model.to(dtype=torch.float16 if args.precision == "fp16" else torch.float32).to(args.device).eval()
    metadata = {"checkpoint": args.checkpoint, "device": args.device, "dtype": args.precision, "torch": torch.__version__,
                "python": platform.python_version(), "threads": args.threads, "batch_size": args.batch_size, "items": len(items),
                "load_seconds": time.perf_counter() - started,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    if args.device == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name()
    if args.output.exists():
        metadata["resumed_rows"] = len(args.output.read_text().splitlines())
    metadata_path = args.output.with_suffix(".meta.json")
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        metadata["previous_invocations"] = previous.pop("previous_invocations", []) + [previous]
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"stage": "loaded", **metadata}), flush=True)
    if args.verify_batch:
        metadata["batch_verification"] = verify_batch(model, items, args.batch_size)
        args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"stage": "batch_verified", **metadata["batch_verification"]}), flush=True)
    if args.verify_only:
        return
    if args.phase == "audit":
        audit(model, items, args.output)
    else:
        run(model, items, args.output, args.batch_size)
    print(json.dumps({"stage": "complete", "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
