"""Evaluate Laya using the exact published BoolQ request template."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from system_one import Laya


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if args.output.exists():
        raise ValueError("Use a fresh output file.")
    items = [json.loads(line) for line in args.items.read_text().splitlines()]
    items = [item for item in items if item["dataset"] == "boolq"]
    if len(items) != 200:
        raise ValueError("Expected all 200 fixed BoolQ cases.")
    adapter = Laya(args.checkpoint, "cpu", torch.float32)
    questions = {"answer": {"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}}
    # The audit helper expects q as its identifier; question IDs are not encoded.
    audit_questions = {"q": questions["answer"]}
    metadata = {"model": "laya", "device": "cpu", "precision": "fp32", "batch_size": 4,
                "items_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source": "NandhaKishorM/laya@d74e8be46ef5241e4ba2b16cd25e896f24f71373/research/scripts/build_benchmark_nb.py",
                "template": "en.boolq", "training_overlap": "The author identifies BoolQ as an in-task retention evaluation.",
                "native_config": adapter.agent.cfg, "verification": []}
    meta = args.output.with_suffix(".meta.json")
    meta.write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.perf_counter()
    with args.output.open("w") as target:
        for offset in range(0, len(items), 4):
            group = items[offset:offset + 4]
            states = [{"passage": item["text"], "question": item["question"]} for item in group]
            audits = []
            for state in states:
                # Match the public SDK's dictionary serialization, including spaces.
                serialized = json.dumps(state, ensure_ascii=False)
                _, audit = adapter.prepare({"state": serialized, "questions": audit_questions})
                audits.append(audit)
            outputs = adapter.agent.predict_batch(states, questions, batch_size=4)
            if offset == 0:
                single = adapter.agent.predict(states[0], questions)["answers"]["answer"]["noul"]
                if (single >= 0.5) != (outputs[0]["answers"]["answer"]["noul"] >= 0.5):
                    raise ValueError("Public batch and single predictions differ.")
                metadata["verification"].append({"batch_matches_single": True, "id": group[0]["id"]})
            for item, output, audit in zip(group, outputs, audits, strict=True):
                prediction = "yes" if output["answers"]["answer"]["noul"] >= 0.5 else "no"
                row = {key: item[key] for key in ("id", "suite", "dataset", "group", "category", "gold")}
                row.update(status="ok", prediction=prediction, correct=prediction == item["gold"],
                           audit=audit, candidate_count=2)
                target.write(json.dumps(row) + "\n")
            target.flush()
            if (offset + len(group)) % 100 == 0:
                print(json.dumps({"model": "laya-published-boolq", "done": offset + len(group),
                                  "seconds": time.perf_counter() - started}), flush=True)
    metadata.update(complete=True, count=len(items), elapsed_seconds=time.perf_counter() - started)
    meta.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
