"""Hold examples and checkpoints fixed while changing decision representations."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from legacy_schemas import load_gliformer as load_model
from legacy_schemas import choice_labels, state_text
from system_one import GlinerDecide, Laya


TASKS = {"jeff/sst2": "sentiment", "jeff/ag_news": "topic", "jeff/emotion": "emotion",
         "computer_science": "answer"}
# First-seen label order in SetFit/ag_news test, as used by the official evaluator.
NEWS_LABELS = {"business": "Business", "sci_tech": "Sci/Tech", "sports": "Sports", "world": "World"}


def variants(item):
    names = ["current", "bare_labels", "short_task", "native"]
    if item["dataset"] == "jeff/sst2":
        names.insert(1, "binary_choice")
    if item["dataset"] == "computer_science":
        names.append("plain_question")
    return names


def represent(item, variant):
    request = copy.deepcopy(item["request"])
    question = request["questions"]["q"]
    backmap = None
    if variant == "binary_choice":
        question.update(type="choice", criteria={"false": "No", "true": "Yes"})
        backmap = {"false": False, "true": True}
    elif variant in ("bare_labels", "short_task", "native"):
        if item["dataset"] == "jeff/sst2":
            question.update(type="choice", criteria={"negative": None, "positive": None})
            backmap = {"negative": False, "positive": True}
        elif item["dataset"] == "computer_science":
            original = question["criteria"]
            labels = [description or key for key, description in original.items()]
            if len(set(labels)) != len(labels):
                raise ValueError("Answer texts must be unique for the bare-label transformation.")
            backmap = dict(zip(labels, original, strict=True))
            question["criteria"] = dict.fromkeys(labels)
        else:
            question["criteria"] = dict.fromkeys(question["criteria"])
        if variant in ("short_task", "native"):
            question["instructions"] = TASKS[item["dataset"]]
        if variant == "native" and item["dataset"] == "jeff/ag_news":
            question["criteria"] = dict.fromkeys(NEWS_LABELS.values())
            backmap = {value: key for key, value in NEWS_LABELS.items()}
    elif variant == "plain_question":
        request["state"] = request["state"]["question"]
    return {"id": item["id"], "dataset": item["dataset"], "gold": item["gold"],
            "variant": variant, "request": request, "backmap": backmap,
            "native": variant == "native", "task": TASKS[item["dataset"]]}


def answer(record, prediction, audit, probability=None):
    if record["backmap"] is not None:
        prediction = record["backmap"][prediction]
    return {"id": record["id"], "dataset": record["dataset"], "variant": record["variant"],
            "gold": record["gold"], "prediction": prediction, "correct": prediction == record["gold"],
            "audit": audit, "probability": probability}


class Gliformer:
    def __init__(self, checkpoint):
        self.model = load_model(checkpoint, "cpu")

    def infer(self, records):
        from gliformer.gliformer import resolve_gliformer_collator_class
        from torch.utils.data import DataLoader

        words, _, _ = self.model.prepare_inputs([state_text(r["request"]["state"]) for r in records])
        inputs = []
        for record, tokens in zip(records, words, strict=True):
            question = record["request"]["questions"]["q"]
            labels = choice_labels(question["criteria"]) if question["type"] == "choice" else [question["instructions"]]
            classes = labels if record["native"] or question["type"] == "noul" else {question["instructions"]: labels}
            inputs.extend(self.model._build_inference_input([tokens], classes=classes))
        cls = self.model.data_collator_class or resolve_gliformer_collator_class(self.model.config)
        collator = cls(self.model.config, data_processor=self.model.data_processor, return_tokens=True, prepare_labels=False)
        batch = collator(inputs)
        lengths = batch["attention_mask"].sum(-1).tolist()
        if any(length > 1024 for length in lengths):
            raise ValueError("A GLiFormer sequence exceeds the experiment's 1,024-token budget.")
        loader = DataLoader(inputs, batch_size=len(records), shuffle=False, collate_fn=collator)
        native = records[0]["native"]
        decoded, _ = self.model._process_multitask_batches(
            loader, threshold=1e-6 if native else -1.0, flat_ner=True, multi_label=not native)
        predictions = self.model._collapse_single_group_results(decoded["classification"])
        result = []
        for record, entries, length in zip(records, predictions, lengths, strict=True):
            question = record["request"]["questions"]["q"]
            scores = {entry["class_name"]: float(entry["score"]) for entry in entries}
            if question["type"] == "noul":
                probability = scores[question["instructions"]]
                prediction = probability >= 0.5
            else:
                labels = choice_labels(question["criteria"])
                label = max(scores, key=scores.get)
                prediction = list(question["criteria"])[labels.index(label)]
                probability = scores[label]
            result.append(answer(record, prediction, {"tokens": length, "state_truncated": False, "head_truncated": False}, probability))
        return result

    def verify_native(self, record, result):
        labels = choice_labels(record["request"]["questions"]["q"]["criteria"])
        values = self.model.classify(state_text(record["request"]["state"]), labels,
                                     threshold=1e-6, multi_label=False, batch_size=1)
        best = max(values, key=lambda value: value["score"])["class_name"]
        key = list(record["request"]["questions"]["q"]["criteria"])[labels.index(best)]
        expected = record["backmap"][key] if record["backmap"] is not None else key
        if expected != result["prediction"]:
            raise ValueError("Native GLiFormer adapter differs from its official classify call.")


class Gliner:
    def __init__(self, checkpoint):
        self.adapter = GlinerDecide(checkpoint, "cpu", torch.float32)
        self.model = self.adapter.model

    def schema(self, record):
        q = record["request"]["questions"]["q"]
        labels = q["criteria"] if q["type"] == "choice" else {"false": "No", "true": "Yes"}
        if record["native"]:
            schema = self.model.create_schema().classification(record["task"], list(labels), class_act="softmax")
        else:
            labels = {key: description or key for key, description in labels.items()}
            schema = self.model.create_schema().classification("q", labels, prompt=q["instructions"], class_act="softmax")
        return schema, list(labels)

    def infer(self, records):
        schemas = [self.schema(record) for record in records]
        batch = self.model.processor.collate_fn_inference(
            [(state_text(r["request"]["state"]), schema) for r, (schema, _) in zip(records, schemas, strict=True)],
            max_len=None, error_policy="raise")
        lengths = batch.attention_mask.sum(-1).tolist()
        if any(length > 1024 for length in lengths):
            raise ValueError("A GLiNER sequence exceeds the experiment's 1,024-token budget.")
        encoded = self.model.encode_tokens(batch)
        _, embeddings = self.model.processor.extract_embeddings_from_batch(encoded, batch.input_ids, batch)
        results = []
        for i, (record, (_, labels)) in enumerate(zip(records, schemas, strict=True)):
            _, _, probabilities = self.model._compute_classification_probabilities(
                batch.original_schemas[i], torch.stack(embeddings[i][0]), batch.schema_tokens_list[i][0])
            scores = dict(zip(labels, probabilities.tolist(), strict=True))
            q = record["request"]["questions"]["q"]
            prediction = scores["true"] >= 0.5 if q["type"] == "noul" else max(scores, key=scores.get)
            results.append(answer(record, prediction, {"tokens": lengths[i], "state_truncated": False, "head_truncated": False}, max(scores.values())))
        return results

    def verify_native(self, record, result):
        schema, _ = self.schema(record)
        value = self.model.extract(state_text(record["request"]["state"]), schema)[record["task"]]
        expected = record["backmap"][value] if record["backmap"] is not None else value
        if expected != result["prediction"]:
            raise ValueError("Native GLiNER adapter differs from its public extraction call.")


class LayaModel:
    def __init__(self, checkpoint):
        self.adapter = Laya(checkpoint, "cpu", torch.float32)
        self.model = self.adapter.model

    def infer(self, records):
        from laya.common import collate_items

        prepared = [self.adapter.prepare(record["request"]) for record in records]
        batch = collate_items([entry[0][1] for entry in prepared], self.adapter.agent.tok.pad_token_id)
        logits, actions = self.adapter.agent._forward(batch)
        results = []
        for i, (record, ((_, items, internal), audit)) in enumerate(zip(records, prepared, strict=True)):
            value = self.adapter.agent._decode_answers(logits, actions, items, ["q"], internal, i)["q"]
            prediction = value["noul"] >= 0.5 if value["type"] == "noul" else value["choice"]
            probability = value["noul"] if value["type"] == "noul" else max(value["probabilities"].values())
            results.append(answer(record, prediction, audit, probability))
        return results

    def verify_native(self, record, result):
        value = self.adapter.agent.predict(state_text(record["request"]["state"]), record["request"]["questions"])["answers"]["q"]
        key = value["choice"]
        expected = record["backmap"][key] if record["backmap"] is not None else key
        if expected != result["prediction"]:
            raise ValueError("Native Laya adapter differs from its public predict call.")


def sample(items, subjects):
    selected = []
    for item in items:
        if item["dataset"] in ("jeff/sst2", "jeff/ag_news", "jeff/emotion"):
            selected.append(item)
        elif subjects.get(item["id"], {}).get("category") == "computer science":
            item = copy.deepcopy(item)
            item["dataset"] = "computer_science"
            selected.append(item)
    if len(selected) != 615:
        raise ValueError("The fixed sample must contain 200 cases per classification task and 15 CS cases.")
    return selected


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("base", "large", "gliner-decide", "laya"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if args.output.exists():
        raise ValueError("Each experiment must use a fresh output file.")
    items = sample([json.loads(x) for x in args.items.read_text().splitlines()], json.loads(args.subjects.read_text()))
    datasets = list(TASKS)
    constructor = Gliformer if args.model in ("base", "large") else Gliner if args.model == "gliner-decide" else LayaModel
    model = constructor(args.checkpoint)
    metadata = {"model": args.model, "checkpoint": args.checkpoint, "device": "cpu", "precision": "fp32",
                "torch": torch.__version__, "batch_size": args.batch_size, "threads": 4,
                "items_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "verification": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = args.output.with_suffix(".meta.json")
    meta.write_text(json.dumps(metadata, indent=2))
    with args.output.open("w") as target:
        for dataset in datasets:
            cases = [item for item in items if item["dataset"] == dataset]
            if args.smoke:
                cases = cases[:2]
            for variant in variants(cases[0]):
                records = [represent(item, variant) for item in cases]
                first = model.infer(records[:2])
                single = [model.infer([record])[0] for record in records[:2]]
                if [row["prediction"] for row in first] != [row["prediction"] for row in single]:
                    raise ValueError(f"Batch verification failed: {dataset}/{variant}.")
                if variant == "native":
                    model.verify_native(records[0], single[0])
                metadata["verification"].append({"dataset": dataset, "variant": variant, "batch_matches_single": True,
                                                   "native_public_call_checked": variant == "native"})
                started = time.perf_counter()
                for offset in range(0, len(records), args.batch_size):
                    chunk = records[offset:offset + args.batch_size]
                    values = model.infer(chunk)
                    for row in values:
                        target.write(json.dumps(row) + "\n")
                    target.flush()
                print(json.dumps({"model": args.model, "dataset": dataset, "variant": variant,
                                  "cases": len(records), "seconds": time.perf_counter() - started}), flush=True)
                meta.write_text(json.dumps(metadata, indent=2))
    metadata["complete"] = True
    meta.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
