"""Choose a deterministic shared sample after both checkpoints' length audits."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--items", type=Path, default=Path(__file__).resolve().parent / "datasets/items.jsonl")
    parser.add_argument("--per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.items.read_text().splitlines()]
    audits = [{row["id"]: row for row in map(json.loads, (args.results / f"{name}-lengths.jsonl").read_text().splitlines())}
              for name in ("base", "large")]
    expected_ids = {item["id"] for item in items}
    if any(set(audit) != expected_ids for audit in audits):
        raise ValueError("Both complete audits must contain every source item exactly once.")
    grouped = defaultdict(list)
    for item in items:
        grouped[item["dataset"]].append(item)
    selected = []
    counts = []
    for dataset, rows in sorted(grouped.items()):
        eligible = [row for row in rows if max(audit[row["id"]]["tokens"] for audit in audits) <= 1024]
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{dataset}".encode()).digest()[:8], "big")
        picked = random.Random(seed).sample(eligible, min(args.per_dataset, len(eligible)))
        selected.extend(sorted(picked, key=lambda row: row["id"]))
        counts.append({"dataset": dataset, "total": len(rows), "eligible": len(eligible),
                       "over_1024": len(rows) - len(eligible), "selected": len(picked)})
    (args.results / "selection.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    (args.results / "selection-manifest.json").write_text(json.dumps({
        "seed": args.seed, "per_dataset": args.per_dataset, "token_limit": 1024,
        "sampling": "uniform without replacement within each shared eligible dataset",
        "counts": counts, "selected": len(selected),
    }, indent=2))
    print(json.dumps({"selected": len(selected), "datasets": len(counts)}))


if __name__ == "__main__":
    main()
