"""Test Laya's JSON state wrapper and its published benchmark request template."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from format_ablation import LayaModel, represent, sample


def transform(item, variant):
    record = represent(item, "current")
    record["variant"] = variant
    if variant == "semantic_noul":
        record["request"]["questions"]["q"]["criteria"] = {"false": "negative sentiment", "true": "positive sentiment"}
        return record
    record["request"]["state"] = {"text": record["request"]["state"]}
    if variant == "laya_reference":
        # Match serialize_state(dict) in the native SDK, including its spaces.
        record["request"]["state"] = json.dumps(record["request"]["state"], ensure_ascii=False)
        name = "ag_news" if item["dataset"] == "jeff/ag_news" else "emotion"
        question = record["request"]["questions"]["q"]
        question["instructions"] = f"Which category does this {name.replace('_', ' ')} text belong to?"
        if name == "ag_news":
            question["criteria"] = {
                "world": "international news, politics, conflicts", "sports": "sports, games, athletes",
                "business": "companies, markets, economy", "sci/tech": "science, technology, software, space"}
            record["backmap"] = {key: "sci_tech" if key == "sci/tech" else key for key in question["criteria"]}
        else:
            question["criteria"] = dict.fromkeys(["sadness", "joy", "love", "anger", "fear", "surprise"])
    return record


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if args.output.exists():
        raise ValueError("Use a new output file for each run.")
    items = sample([json.loads(x) for x in args.items.read_text().splitlines()], json.loads(args.subjects.read_text()))
    model = LayaModel(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"device": "cpu", "precision": "fp32", "batch_size": 4, "torch": torch.__version__,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reference": "NandhaKishorM/laya@d74e8be46ef5241e4ba2b16cd25e896f24f71373/benchmarks/bench_fast.py",
                "notes": "Same 200 cases per task; no reference split or published score is being reproduced."}
    meta = args.output.with_suffix(".meta.json")
    meta.write_text(json.dumps(metadata, indent=2))
    with args.output.open("w") as target:
        for dataset in ("jeff/sst2", "jeff/ag_news", "jeff/emotion"):
            for variant in ("json_state", "laya_reference", "semantic_noul"):
                if variant == "laya_reference" and dataset == "jeff/sst2":
                    continue
                if variant == "semantic_noul" and dataset != "jeff/sst2":
                    continue
                records = [transform(item, variant) for item in items if item["dataset"] == dataset]
                paired = model.infer(records[:2])
                single = [model.infer([record])[0] for record in records[:2]]
                if [row["prediction"] for row in paired] != [row["prediction"] for row in single]:
                    raise ValueError("Batch and single-item verification failed.")
                if variant == "laya_reference":
                    first = records[0]
                    native = model.adapter.agent.predict(json.loads(first["request"]["state"]), first["request"]["questions"])["answers"]["q"]["choice"]
                    expected = first["backmap"][native] if first["backmap"] is not None else native
                    if single[0]["prediction"] != expected:
                        raise ValueError("The reference template differs from the SDK's native dictionary-state call.")
                started = time.perf_counter()
                for offset in range(0, len(records), 4):
                    for row in model.infer(records[offset:offset + 4]):
                        target.write(json.dumps(row) + "\n")
                    target.flush()
                print(json.dumps({"model": "laya", "dataset": dataset, "variant": variant,
                                  "cases": len(records), "seconds": time.perf_counter() - started}), flush=True)
    metadata["complete"] = True
    meta.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
