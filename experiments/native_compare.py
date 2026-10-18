"""Compare fixed checkpoints through their public native classification methods."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from legacy_schemas import load_gliformer as load_model
from system_one import GlinerDecide, Laya


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    root = args.data
    items = []
    omitted = []
    for source in sorted((root / "fast-decisions").glob("*.jsonl")):
        domain = source.stem
        for index, line in enumerate(source.read_text().splitlines()):
            row = json.loads(line)
            heads = row["output"]["classifications"]
            # One single-label decision per document keeps domains equally weighted.
            position, head = next((i, h) for i, h in enumerate(heads) if not h.get("multi_label", False))
            if len(head["true_label"]) != 1 or len(set(head["labels"])) != len(head["labels"]):
                raise ValueError("Expected one gold label and distinct candidates.")
            items.append({"id": f"fast-decisions/{domain}/{index}", "suite": "operations",
                          "dataset": domain, "group": f"{domain}/{index}",
                          "text": row["input"], "question": None, "task": head["task"],
                          "labels": head["labels"], "gold": head["true_label"][0],
                          "category": None, "head_index": position})
            omitted.append({"id": items[-1]["id"], "unscored_heads": len(heads) - 1})
    subjects = json.loads((root / "mmlu-pro-subjects.json").read_text())
    for line in (root / "items.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["dataset"] not in ("leanmcp/mmlu_pro", "jeff/boolq"):
            continue
        question = row["request"]["questions"]["q"]
        if row["dataset"] == "leanmcp/mmlu_pro":
            labels = list(question["criteria"].values())
            items.append({"id": row["id"], "suite": "knowledge", "dataset": "mmlu_pro",
                          "group": row["group"], "text": row["request"]["state"]["question"],
                          "question": "Select the best answer.", "task": "answer", "labels": labels,
                          "gold": question["criteria"][row["gold"]],
                          "category": subjects[row["id"]]["category"]})
        else:
            items.append({"id": row["id"], "suite": "reading", "dataset": "boolq",
                          "group": row["group"], "text": row["request"]["state"]["passage"],
                          "question": question["instructions"], "task": "answer",
                          "labels": ["yes", "no"], "gold": "yes" if row["gold"] else "no",
                          "category": None})
    if len(items) != 2400 or len({r["id"] for r in items}) != len(items):
        raise ValueError("Expected 1,700 operational, 500 MMLU-Pro, and 200 BoolQ cases.")
    for item in items:
        if len(set(item["labels"])) != len(item["labels"]) or item["gold"] not in item["labels"]:
            raise ValueError("Invalid candidate set.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    (args.output / "omitted-heads.json").write_text(json.dumps(omitted, indent=2) + "\n")
    protocol = {
        "items": len(items), "datasets": dict(Counter(r["dataset"] for r in items)),
        "models": ["base", "large", "gliner-decide", "laya"],
        "device": "cpu", "precision": "fp32", "threads": 4, "batch_size": 4,
        "operations": "All 100 public development documents per domain; first single-label head per document. No multi-label conversion. Published held-out test scores are not reproduced.",
        "knowledge": "All 500 pinned MMLU-Pro cases, with complete answer texts as candidates; report subjects including computer science separately.",
        "reading": "All 200 pinned Jeff BoolQ cases, with yes/no candidates and the original question.",
        "gliformer": "Public classify; bare candidates for exams, named task group for operational decisions, and passage/question prose for BoolQ. Single-label, threshold 1e-6.",
        "gliner": "Public classify_text/batch_classify_text; original task name, bare labels, multi_label=False, question prompt when applicable.",
        "laya": "Public predict/predict_batch; raw text state and a typed choice question. Original question or readable task name, bare candidates. Shipped context/head budgets are retained.",
        "context": "Audit complete prepared lengths. Skip GLiFormer/GLiNER inputs above 1,024 tokens. Record all native Laya truncation; primary paired comparisons use the common untruncated subset.",
        "selection": "No prompt tuning, checkpoint switching, or per-task winning format selection. Every model starts a fresh evaluation.",
        "scope": "Official SDK methods with a fixed documented benchmark mapping; not a claim that every model has a published official adapter for these datasets.",
        "sources": {"normalized_items_sha256": sha(root / "items.jsonl"),
                    "subjects_sha256": sha(root / "mmlu-pro-subjects.json"),
                    "fast_decisions": json.loads((root / "fast-decisions/manifest.json").read_text())},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(args.output)


class NativeModel:
    def __init__(self, name, checkpoint):
        self.name = name
        if name in ("base", "large"):
            self.model = load_model(checkpoint, "cpu")
            self.model.config.max_len = 1024
            self.model.data_processor.config.max_len = 1024
        elif name == "gliner-decide":
            self.model = GlinerDecide(checkpoint, "cpu", torch.float32).model
        else:
            self.adapter = Laya(checkpoint, "cpu", torch.float32)
            self.model = self.adapter.model

    def payload(self, item):
        if self.name in ("base", "large"):
            text = item["text"]
            if item["dataset"] == "boolq":
                text = f"Passage:\n{text}\n\nQuestion:\n{item['question']}\nAnswer:"
            classes = {item["task"]: item["labels"]} if item["suite"] == "operations" else item["labels"]
            return text, classes
        if self.name == "gliner-decide":
            task = {"labels": item["labels"], "multi_label": False}
            if item["question"]:
                task["prompt"] = item["question"]
            return item["text"], {item["task"]: task}
        question = {"type": "choice", "instructions": item["question"] or item["task"].replace("_", " "),
                    "criteria": dict.fromkeys(item["labels"])}
        return item["text"], {"q": question}

    def audit(self, item):
        text, task = self.payload(item)
        if self.name in ("base", "large"):
            from gliformer.gliformer import resolve_gliformer_collator_class

            old = self.model.config.max_len
            try:
                self.model.config.max_len = 1_000_000
                self.model.data_processor.config.max_len = 1_000_000
                words, _, _ = self.model.prepare_inputs([text])
                inputs = self.model._build_inference_input(words, classes=task)
                cls = self.model.data_collator_class or resolve_gliformer_collator_class(self.model.config)
                batch = cls(self.model.config, data_processor=self.model.data_processor,
                            return_tokens=True, prepare_labels=False)(inputs)
                length = int(batch["attention_mask"].sum().item())
            finally:
                self.model.config.max_len = old
                self.model.data_processor.config.max_len = old
            return {"tokens": length, "token_limit": 1024, "over_limit": length > 1024,
                    "state_truncated": False, "head_truncated": False}
        if self.name == "gliner-decide":
            schema = self.model._classification_schema(task)
            batch = self.model.processor.collate_fn_inference([(text, schema)], max_len=None, error_policy="raise")
            length = int(batch.attention_mask.sum().item())
            return {"tokens": length, "token_limit": 1024, "over_limit": length > 1024,
                    "state_truncated": False, "head_truncated": False}
        _, audit = self.adapter.prepare({"state": text, "questions": task})
        return audit

    def predict(self, items, batch=True):
        payloads = [self.payload(item) for item in items]
        task = payloads[0][1]
        if any(p[1] != task for p in payloads):
            raise ValueError("A public batch call must share one candidate schema.")
        texts = [p[0] for p in payloads]
        if self.name in ("base", "large"):
            outputs = self.model.classify(texts=texts if batch else texts[0], classes=task,
                                          multi_label=False, threshold=1e-6, batch_size=4 if batch else 1)
            if not batch:
                outputs = [outputs]
            return [max(value, key=lambda x: x["score"])["class_name"] for value in outputs]
        if self.name == "gliner-decide":
            if batch:
                outputs = self.model.batch_classify_text(texts, task, batch_size=4)
            else:
                outputs = [self.model.classify_text(texts[0], task)]
            return [value[items[0]["task"]] for value in outputs]
        if batch:
            outputs = self.adapter.agent.predict_batch(texts, task, batch_size=4)
        else:
            outputs = [self.adapter.agent.predict(texts[0], task)]
        return [value["answers"]["q"]["choice"] for value in outputs]


def schema_key(model, item):
    return json.dumps(model.payload(item)[1], ensure_ascii=False)


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if args.output.exists():
        raise ValueError("Use a fresh output path for every run.")
    items = [json.loads(s) for s in args.items.read_text().splitlines()]
    if args.smoke:
        # Include every operational schema and both question-answering suites.
        selected = {}
        for item in items:
            key = item["dataset"]
            if key not in selected:
                selected[key] = item
        items = list(selected.values())
    model = NativeModel(args.model, args.checkpoint)
    metadata = {"model": args.model, "checkpoint": args.checkpoint, "device": "cpu", "precision": "fp32",
                "torch": torch.__version__, "python": platform.python_version(), "batch_size": 4,
                "items_sha256": sha(args.items), "script_sha256": sha(Path(__file__)),
                "parameters": sum(p.numel() for p in model.model.parameters()), "verification": [],
                "smoke": args.smoke, "public_methods_only_for_predictions": True}
    if args.model == "laya":
        metadata["native_config"] = model.adapter.agent.cfg
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    verified = set()
    index = 0
    errors = 0
    started = time.perf_counter()
    with args.output.open("w") as target:
        while index < len(items):
            first = items[index]
            group = [first]
            key = schema_key(model, first)
            while index + len(group) < len(items) and len(group) < 4:
                candidate = items[index + len(group)]
                if schema_key(model, candidate) != key:
                    break
                group.append(candidate)
            audited = [(item, model.audit(item)) for item in group]
            supported = [(item, audit) for item, audit in audited if not audit["over_limit"]]
            result = {}
            if supported:
                examples = [item for item, _ in supported]
                tick = time.perf_counter()
                try:
                    predictions = model.predict(examples)
                    if len(predictions) != len(examples):
                        raise ValueError("Wrong number of native predictions.")
                    if first["dataset"] not in verified:
                        single = model.predict([examples[0]], batch=False)[0]
                        if single != predictions[0]:
                            raise ValueError("Public batch and single-item decisions differ.")
                        metadata["verification"].append({"dataset": first["dataset"], "id": examples[0]["id"],
                                                         "batch_matches_single": True})
                        verified.add(first["dataset"])
                    elapsed = (time.perf_counter() - tick) / len(examples)
                    for item, prediction in zip(examples, predictions, strict=True):
                        if prediction not in item["labels"]:
                            raise ValueError("Native prediction is not a candidate label.")
                        result[item["id"]] = {"status": "ok", "prediction": prediction,
                                              "correct": prediction == item["gold"], "seconds": elapsed}
                except Exception as exc:
                    errors += 1
                    for item in examples:
                        result[item["id"]] = {"status": "error", "error": str(exc), "error_type": type(exc).__name__}
            for item, audit in audited:
                row = {k: item[k] for k in ("id", "suite", "dataset", "group", "category", "gold")}
                row.update(audit=audit, candidate_count=len(item["labels"]))
                row.update(result.get(item["id"], {"status": "unsupported", "reason": "prepared_length_exceeds_1024"}))
                target.write(json.dumps(row) + "\n")
            target.flush()
            index += len(group)
            if index % 100 < len(group) or index == len(items):
                print(json.dumps({"model": args.model, "done": index, "total": len(items),
                                  "seconds": time.perf_counter() - started, "errors": errors}), flush=True)
            if errors:
                raise RuntimeError("Stopped after a native inference or verification error; inspect the output.")
    metadata.update(complete=True, count=len(items), elapsed_seconds=time.perf_counter() - started)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--data", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--model", choices=("base", "large", "gliner-decide", "laya"), required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--items", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
