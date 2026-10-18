"""Compare native English System One inference on the fixed quality sample."""

import argparse
import gc
import hashlib
import json
import math
import platform
import resource
import time
import warnings
from pathlib import Path

import torch

from legacy_schemas import state_text


class UnsupportedInput(ValueError):
    pass


class GlinerDecide:
    def __init__(self, checkpoint, device, dtype):
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained(str(checkpoint), attention_backend="standard")
        self.model.to(device=device, dtype=dtype).eval()
        self.device = device

    def prepare(self, request):
        question = request["questions"]["q"]
        labels = (question["criteria"] if question["type"] == "choice"
                  else {"false": "No", "true": "Yes"})
        labels = {key: description or key for key, description in labels.items()}
        schema = self.model.create_schema().classification(
            "q", labels, prompt=question["instructions"], class_act="softmax")
        batch = self.model.processor.collate_fn_inference(
            [(state_text(request["state"]), schema)], max_len=None, error_policy="raise")
        tokens = batch.schema_tokens_list[0][0]
        encoded_labels = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "[L]"]
        if encoded_labels != list(labels):
            raise ValueError("Encoded label order differs from the requested label order.")
        length = int(batch.attention_mask.sum().item())
        audit = {"tokens": length, "token_limit": 1024, "state_truncated": False,
                 "head_truncated": False, "over_limit": length > 1024}
        return (batch, list(labels), question["type"]), audit

    def infer(self, prepared):
        batch, labels, kind = prepared
        batch = batch.to(self.device)
        encoded = self.model.encode_tokens(batch)
        _, schemas = self.model.processor.extract_embeddings_from_batch(encoded, batch.input_ids, batch)
        # Use the public runtime's scoring implementation, including its FP16 softmax.
        _, _, probabilities = self.model._compute_classification_probabilities(
            batch.original_schemas[0], torch.stack(schemas[0][0]), batch.schema_tokens_list[0][0])
        probabilities = probabilities.tolist()
        scores = dict(zip(labels, probabilities, strict=True))
        if kind == "noul":
            return {"type": "noul", "noul": scores["true"], "probabilities": scores}
        return {"type": "choice", "choice": max(scores, key=scores.get), "probabilities": scores}

    def verify(self, request, answer):
        question = request["questions"]["q"]
        labels = question["criteria"] if question["type"] == "choice" else {"false": "No", "true": "Yes"}
        labels = {key: description or key for key, description in labels.items()}
        result = self.model.classify_text(state_text(request["state"]), {
            "q": {"labels": labels, "prompt": question["instructions"], "class_act": "softmax"}},
            include_confidence=True)["q"]
        scores = answer["probabilities"]
        expected = max(scores, key=scores.get)
        score = scores[expected]
        if result["label"] != expected or abs(result["confidence"] - score) > 1e-5:
            raise ValueError(f"The benchmark adapter differs from the public classification API: "
                             f"adapter_label={expected!r}, public_label={result['label']!r}, "
                             f"adapter_score={score}, public_score={result['confidence']}, "
                             f"absolute_delta={abs(result['confidence'] - score)}.")


class Laya:
    def __init__(self, checkpoint, device, dtype):
        import laya

        self.agent = laya.load(str(checkpoint), device="cpu", backend="eager")
        self.model = self.agent.model.to(device=device, dtype=dtype).eval()
        self.agent.device = torch.device(device)
        self.agent.dtype = dtype
        self.agent.amp_enabled = device == "cuda"
        # Fail the benchmark request instead of silently counting a CPU retry as GPU work.
        warnings.filterwarnings("error", message="Warning: GPU memory exceeded during inference.*")

    def prepare(self, request):
        from laya.common import build_head, collate_items, encode_text, render_options

        question = request["questions"]["q"]
        internal = {"q": self.agent._to_internal(question)}
        state = state_text(request["state"])
        items = self.agent._encode_state(state, ["q"], internal)
        item = items[0]
        full_head, _, _ = build_head(self.agent.tok, internal["q"], head_max_len=1_000_000)
        used_head, _, _ = build_head(self.agent.tok, internal["q"], self.agent.cfg["head_max_len"])
        option_cap = any(len(encode_text(self.agent.tok, " " + option, add_special_tokens=False)["input_ids"]) > 48
                         for option in render_options(internal["q"]))
        audit = {"tokens": len(item["ids"]), "token_limit": self.agent.cfg["max_len"],
                 "state_truncated": item["state_stats"]["truncated"],
                 "state_tokens_dropped": item["state_stats"]["state_tokens_dropped"],
                 "head_truncated": len(full_head) > len(used_head) or option_cap,
                 "options_distinct": item["options"]["options_distinct"], "over_limit": False}
        return (collate_items([items], self.agent.tok.pad_token_id), items, internal), audit

    def infer(self, prepared):
        batch, items, internal = prepared
        logits, actions = self.agent._forward(batch)
        return self.agent._decode_answers(logits, actions, items, ["q"], internal, 0)["q"]

    def verify(self, request, answer):
        result = self.agent.predict(state_text(request["state"]), request["questions"])["answers"]["q"]
        if result != answer:
            raise ValueError("The benchmark adapter differs from the public Laya API.")


def prediction_row(item, answer, error, audit, seconds):
    row = {key: item[key] for key in ("id", "dataset", "gold", "group")}
    row.update(seconds=seconds, audit=audit, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    if error:
        row.update(status="unsupported" if error["error_type"] == "UnsupportedInput" else "error", **error)
        return row
    if answer["type"] == "noul":
        probability = answer["noul"]
        prediction = probability >= 0.5
        scores = {"true": probability, "false": 1 - probability}
        gold_key = "true" if item["gold"] else "false"
    else:
        prediction = answer["choice"]
        scores = answer["probabilities"]
        gold_key = item["gold"]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("The model produced invalid probabilities.")
    row.update(status="ok", answer=answer, prediction=prediction, correct=prediction == item["gold"],
               confidence=max(scores.values()),
               brier=sum((score - int(label == gold_key)) ** 2 for label, score in scores.items()))
    return row


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gliner-decide", "laya"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.output.exists():
        raise ValueError("Use a new output path to keep each benchmark run independent.")
    items = [json.loads(line) for line in args.items.read_text().splitlines()]
    if args.limit:
        # Include all task families in short compatibility runs.
        grouped = {}
        for item in items:
            grouped.setdefault(item["dataset"], item)
        items = list(grouped.values())[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name("system_one_models.json").read_text())
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    started = time.perf_counter()
    adapter = (GlinerDecide if args.model == "gliner-decide" else Laya)(args.checkpoint, args.device, dtype)
    synchronize(args.device)
    metadata = {"model": args.model, "checkpoint": manifest["models"][args.model], "sources": manifest["sources"],
                "python": platform.python_version(), "torch": torch.__version__, "device": args.device,
                "precision": args.precision, "parameter_count": sum(p.numel() for p in adapter.model.parameters()),
                "load_seconds": time.perf_counter() - started, "items": len(items), "batch_size": 1,
                "items_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "noul_mapping": "native binary false/true softmax"}
    if args.model == "laya":
        metadata["native_config"] = adapter.agent.cfg
        metadata["applied_temperatures"] = adapter.agent.temperature
        metadata["applied_temperatures_by_options"] = adapter.agent.temperature_by_options
    metadata_path = args.output.with_suffix(".meta.json")
    if args.device == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name()
        metadata["compute_capability"] = torch.cuda.get_device_capability()
        metadata["loaded_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"stage": "loaded", "model": args.model, "parameters": metadata["parameter_count"]}), flush=True)
    if args.verify:
        verified = []
        for kind in ("choice", "noul"):
            item = next(item for item in items if item["request"]["questions"]["q"]["type"] == kind)
            prepared, audit = adapter.prepare(item["request"])
            if audit["over_limit"]:
                raise ValueError("The verification example exceeds the experiment budget.")
            answer = adapter.infer(prepared)
            adapter.verify(item["request"], answer)
            verified.append(item["id"])
        metadata["public_api_verified"] = verified
        metadata_path.write_text(json.dumps(metadata, indent=2))
    # Warm up with the first request. Keep its memory in the global peak, but not its latency.
    prepared, audit = adapter.prepare(items[0]["request"])
    if not audit["over_limit"]:
        adapter.infer(prepared)
    synchronize(args.device)
    del prepared
    with args.output.open("w") as target:
        consecutive_errors = 0
        for index, item in enumerate(items):
            audit, answer, error, prepared = {}, None, None, None
            synchronize(args.device)
            start = time.perf_counter()
            try:
                prepared, audit = adapter.prepare(item["request"])
                if audit["over_limit"]:
                    raise UnsupportedInput("The fully prepared sequence exceeds the 1,024-token experiment budget.")
                answer = adapter.infer(prepared)
                synchronize(args.device)
                consecutive_errors = 0
            except Exception as caught:
                error = {"error_type": type(caught).__name__, "error": str(caught)}
                consecutive_errors = 0 if isinstance(caught, UnsupportedInput) else consecutive_errors + 1
            elapsed = time.perf_counter() - start
            row = prediction_row(item, answer, error, audit, elapsed)
            if args.device == "cuda":
                row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            target.write(json.dumps(row) + "\n")
            target.flush()
            del prepared
            if error:
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            if (index + 1) % 100 == 0:
                print(json.dumps({"stage": "evaluate", "done": index + 1, "total": len(items)}), flush=True)
            if consecutive_errors >= 3:
                raise RuntimeError("Stopped after three consecutive inference errors.")
    if args.device == "cuda":
        metadata["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        metadata["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"stage": "complete", "model": args.model, "items": len(items)}), flush=True)


if __name__ == "__main__":
    main()
