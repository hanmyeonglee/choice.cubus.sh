"""Compare precision, encoder quantization, and explicit CPU retry in isolation."""

import argparse
import gc
import json
import math
import os
import resource
import time
from pathlib import Path

import torch

from legacy_schemas import load_gliformer as load_model
from legacy_schemas import EvaluateRequest, choice_labels, state_text


def is_cuda_oom(error):
    # TorchScript can wrap CUDA OOM in RuntimeError instead of OutOfMemoryError.
    return isinstance(error, torch.OutOfMemoryError) or "cuda out of memory" in str(error).lower()


def memory():
    result = {"peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info()
        result.update(
            allocated_mib=torch.cuda.memory_allocated() / 1024**2,
            reserved_mib=torch.cuda.memory_reserved() / 1024**2,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
            device_used_mib=(total - free) / 1024**2,
            device_total_mib=total / 1024**2,
        )
    return result


def quantize_encoder(model, compute_dtype):
    import bitsandbytes as bnb

    encoder = model.model.token_rep_layer.bert_layer.model.encoder
    count = 0

    def replace(parent):
        nonlocal count
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                layer = bnb.nn.Linear4bit(
                    child.in_features, child.out_features, bias=child.bias is not None,
                    compute_dtype=compute_dtype, compress_statistics=True, quant_type="nf4",
                )
                layer.weight = bnb.nn.Params4bit(
                    child.weight.detach(), requires_grad=False,
                    compress_statistics=True, quant_type="nf4",
                )
                if child.bias is not None:
                    layer.bias = torch.nn.Parameter(child.bias.detach(), requires_grad=False)
                setattr(parent, name, layer)
                count += 1
            else:
                replace(child)

    replace(encoder)
    return {"bitsandbytes": bnb.__version__, "quantized_encoder_linear_layers": count}


def evaluate(model, body):
    answers = {}
    with torch.inference_mode():
        for name, question in body["questions"].items():
            labels = choice_labels(question["criteria"]) if question["type"] == "choice" else [question["instructions"]]
            classes = {question["instructions"]: labels} if question["type"] == "choice" else labels
            predictions = model.classify(state_text(body["state"]), classes, threshold=-1.0, multi_label=True, batch_size=1)
            scores = {item["class_name"]: float(item["score"]) for item in predictions}
            if any(not math.isfinite(scores[label]) for label in labels):
                raise ValueError("The model returned non-finite classification scores.")
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": scores[labels[0]]}
            else:
                total = sum(scores[label] for label in labels)
                if not math.isfinite(total) or total <= 0:
                    raise ValueError("The model returned an invalid choice score total.")
                probabilities = {key: scores[label] / total for key, label in zip(question["criteria"], labels)}
                answers[name] = {"type": "choice", "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}
    return answers


def attempt(model, body, device):
    started = time.perf_counter()
    try:
        answers = evaluate(model, body)
        if device == "cuda":
            torch.cuda.synchronize()
        return {"status": "ok", "answers": answers, "device": device, "seconds": time.perf_counter() - started, "memory": memory()}
    except Exception as error:
        return {
            "status": "oom" if is_cuda_oom(error) else "error", "device": device,
            "error_type": type(error).__name__, "error": str(error),
            "seconds": time.perf_counter() - started, "memory": memory(),
        }


def quality(answers, case):
    checks = []
    for name, expected in case.get("expected_choices", {}).items():
        actual = answers.get(name, {}).get("choice")
        checks.append({"question": name, "type": "choice", "expected": expected, "actual": actual, "correct": actual == expected})
    for name, expected in case.get("expected_noul", {}).items():
        score = answers.get(name, {}).get("noul")
        checks.append({"question": name, "type": "noul", "expected": expected, "score": score, "correct_at_0_5": score is not None and (score >= 0.5) == expected})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--memory-mode", choices=("gpu", "cpu-fallback"), default="gpu")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cases", type=Path, default=Path("/probe/cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-filter", default="")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"name": args.name, "precision": args.precision, "quantization": args.quantization, "memory_mode": args.memory_mode, "checkpoint": args.checkpoint, "cases": [], "completed": False, "torch": torch.__version__, "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    save()
    try:
        device = args.device
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable.")
            report["gpu"] = torch.cuda.get_device_name()
            report["compute_capability"] = list(torch.cuda.get_device_capability())
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model = load_model(args.checkpoint, "cpu")
        report["logical_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        dtype = torch.float16 if args.precision == "fp16" else torch.float32
        model.to(dtype=dtype)
        if args.quantization == "nf4":
            if device != "cuda":
                raise ValueError("The NF4 experiment requires CUDA.")
            report.update(quantize_encoder(model, dtype))
        model.to(device).eval()
        if device == "cuda":
            torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - load_started
        report["after_load"] = memory()
        save()
        print(json.dumps({"stage": "loaded", "name": args.name, "memory": report["after_load"]}), flush=True)
        cases = json.loads(args.cases.read_text())
        for case in cases:
            if args.case_filter and args.case_filter not in case["name"]:
                continue
            EvaluateRequest.model_validate(case["request"])
            report["in_progress"] = {"name": case["name"], "device": device, "started_unix": time.time()}
            save()
            result = attempt(model, case["request"], device)
            result["name"] = case["name"]
            if result["status"] == "oom" and args.memory_mode == "cpu-fallback" and device == "cuda":
                gpu_failure = result
                # Drop all GPU references, then reload FP32 for an explicit CPU retry.
                del model
                gc.collect()
                torch.cuda.empty_cache()
                report["in_progress"].update(stage="cpu_reload", gpu_failure=gpu_failure)
                save()
                reload_started = time.perf_counter()
                model = load_model(args.checkpoint, "cpu").float().eval()
                reload_seconds = time.perf_counter() - reload_started
                device = "cpu"
                report["in_progress"].update(stage="cpu_retry", cpu_reload_seconds=reload_seconds, device=device)
                save()
                result = attempt(model, case["request"], device)
                result.update(name=case["name"], gpu_failure=gpu_failure, cpu_reload_seconds=reload_seconds)
                result["total_with_fallback_seconds"] = gpu_failure["seconds"] + reload_seconds + result["seconds"]
            if result["status"] == "ok":
                result["quality_checks"] = quality(result["answers"], case)
            report["cases"].append(result)
            report.pop("in_progress", None)
            save()
            print(json.dumps({"stage": "case", "name": args.name, "case": case["name"], "status": result["status"], "device": result["device"], "seconds": result["seconds"], "memory": result["memory"]}), flush=True)
            if result["status"] == "oom":
                gc.collect()
                torch.cuda.empty_cache()
        report["completed"] = True
        report["final_memory"] = memory()
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        try:
            report["failure_memory"] = memory()
        except Exception as memory_error:
            report["memory_error"] = str(memory_error)
        print(json.dumps({"stage": "failed", "name": args.name, "error_type": type(error).__name__, "error": str(error)}), flush=True)
    finally:
        save()


if __name__ == "__main__":
    main()
