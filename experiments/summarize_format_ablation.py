"""Summarize complete paired format ablations without selecting a best prompt."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics


def truncated(row):
    audit = row["audit"]
    return audit.get("state_truncated", False) or audit.get("head_truncated", False)


def paired(baseline, variant, ids):
    if not ids:
        return None
    differences = [int(variant[key]["correct"]) - int(baseline[key]["correct"]) for key in ids]
    rng = random.Random(42)
    draws = sorted(statistics.mean(rng.choices(differences, k=len(ids))) for _ in range(2000))
    return {"n": len(ids), "delta": statistics.mean(differences),
            "ci95": [draws[50], draws[1949]],
            "new_correct": differences.count(1), "new_incorrect": differences.count(-1),
            "decisions_changed": sum(variant[key]["prediction"] != baseline[key]["prediction"] for key in ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    report = {"models": {}, "device": "cpu", "precision": "fp32", "paired_examples": True}
    lines = ["# Decision representation ablation", "",
             "Fixed checkpoints and fixed examples, CPU FP32, batch 4, four Torch threads per model. No training, calibration fitting, or gold-based prompt selection.",
             "The same baseline is rerun under the same precision and batching. Published full-dataset scores use different samples and are contextual references.",
             "Variant comparisons are exploratory; confidence intervals are paired bootstrap intervals over these sampled cases and are not adjusted for multiple comparisons.", ""]
    csv_rows = []
    for model in ("base", "large", "gliner-decide", "laya"):
        path = args.results / f"{model}.jsonl"
        metadata = json.loads(path.with_suffix(".meta.json").read_text())
        if not metadata.get("complete"):
            raise ValueError(f"Incomplete {model} evaluation.")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 2675:
            raise ValueError(f"Expected 2,675 rows for {model}; received {len(rows)}.")
        if model == "laya":
            extra = args.results / "laya-reference.jsonl"
            if not json.loads(extra.with_suffix(".meta.json").read_text()).get("complete"):
                raise ValueError("The Laya reference-template run is incomplete.")
            extra_rows = [json.loads(line) for line in extra.read_text().splitlines()]
            if len(extra_rows) != 1200:
                raise ValueError("Expected 1,200 Laya reference-template and semantic-noul results.")
            rows.extend(extra_rows)
        grouped = defaultdict(dict)
        for row in rows:
            group = grouped[(row["dataset"], row["variant"])]
            if row["id"] in group:
                raise ValueError("Duplicate case in a representation condition.")
            group[row["id"]] = row
        model_report = {}
        lines += [f"## {model}", "", "| Dataset | Format | Correct | Accuracy | Delta vs current | Truncated |",
                  "|---|---|---:|---:|---:|---:|"]
        for (dataset, variant), values in grouped.items():
            baseline = grouped[(dataset, "current")]
            ids = sorted(baseline)
            if set(values) != set(baseline):
                raise ValueError("Representation conditions must contain identical case IDs.")
            accuracy = statistics.mean(row["correct"] for row in values.values())
            clean = [key for key in ids if not truncated(values[key]) and not truncated(baseline[key])]
            comparison = paired(baseline, values, ids)
            clean_comparison = paired(baseline, values, clean)
            record = {"n": len(values), "correct": sum(row["correct"] for row in values.values()),
                      "accuracy": accuracy, "truncated": sum(truncated(row) for row in values.values()),
                      "tokens_min": min(row["audit"]["tokens"] for row in values.values()),
                      "tokens_max": max(row["audit"]["tokens"] for row in values.values()),
                      "paired_vs_current": comparison,
                      "untruncated_pair": {"paired": clean_comparison,
                                           "current_accuracy": statistics.mean(baseline[key]["correct"] for key in clean) if clean else None,
                                           "variant_accuracy": statistics.mean(values[key]["correct"] for key in clean) if clean else None}}
            model_report.setdefault(dataset, {})[variant] = record
            delta = f"{comparison['delta'] * 100:+.1f} pp [{comparison['ci95'][0] * 100:+.1f}, {comparison['ci95'][1] * 100:+.1f}]"
            lines.append(f"| {dataset} | {variant} | {record['correct']}/{record['n']} | {accuracy * 100:.1f}% | {delta} | {record['truncated']} |")
            csv_rows.append({"model": model, "dataset": dataset, "variant": variant, "n": record["n"],
                             "correct": record["correct"], "accuracy": accuracy, "truncated": record["truncated"],
                             "delta": comparison["delta"], "ci95_low": comparison["ci95"][0], "ci95_high": comparison["ci95"][1]})
        report["models"][model] = model_report
        lines.append("")
    (args.results / "format-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.results / "format-summary.md").write_text("\n".join(lines))
    with (args.results / "format-summary.csv").open("w") as target:
        writer = csv.DictWriter(target, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(args.results / "format-summary.md")


if __name__ == "__main__":
    main()
