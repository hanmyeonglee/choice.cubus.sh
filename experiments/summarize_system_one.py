"""Report paired native-model quality, input coverage, latency, and GPU memory."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from summarize_benchmark import metrics, paired, percentile


def load(path):
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    items = load(args.items)
    audits = load(args.baseline / "base-lengths.jsonl")
    predictions = {name: load(args.baseline / f"{name}-predictions.jsonl") for name in ("base", "large")}
    runtime = {}
    incomplete = []
    for name in ("gliner-decide", "laya"):
        path = args.results / f"{name}-predictions.jsonl"
        if not path.is_file():
            incomplete.append(name)
            continue
        rows = load(path)
        if set(rows) != set(items):
            incomplete.append(name)
            continue
        predictions[name] = rows
        metadata = json.loads(path.with_suffix(".meta.json").read_text())
        memory = []
        for row in csv.reader((args.results / f"{name}-gpu.csv").read_text().splitlines()):
            try:
                memory.append(float(row[1]))
            except (ValueError, IndexError):
                continue
        valid = [row for row in rows.values() if row["status"] == "ok"]
        runtime[name] = {"parameter_count": metadata["parameter_count"], "precision": metadata["precision"],
                         "p50_ms": percentile([row["seconds"] * 1000 for row in valid], 0.5),
                         "p95_ms": percentile([row["seconds"] * 1000 for row in valid], 0.95),
                         "board_peak_mib": max(memory) if memory else None,
                         "peak_allocated_mib": metadata.get("peak_allocated_mib"),
                         "peak_reserved_mib": metadata.get("peak_reserved_mib"),
                         "status_counts": dict(Counter(row["status"] for row in rows.values())),
                         "state_truncated": sum(row.get("audit", {}).get("state_truncated", False) for row in rows.values()),
                         "head_truncated": sum(row.get("audit", {}).get("head_truncated", False) for row in rows.values())}
    for name in ("base", "large"):
        metadata_path = args.baseline / f"{name}-predictions.meta.json"
        valid = list(predictions[name].values())
        runtime[name] = {"p50_ms": percentile([row["seconds"] * 1000 for row in valid], 0.5),
                         "p95_ms": percentile([row["seconds"] * 1000 for row in valid], 0.95),
                         "metadata": json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}}
    common = sorted(key for key in items if all(rows[key]["status"] == "ok" for rows in predictions.values()))
    untruncated = [key for key in common if all(
        not rows[key].get("audit", {}).get("state_truncated", False)
        and not rows[key].get("audit", {}).get("head_truncated", False) for rows in predictions.values())]
    cohorts = {"all_selected": sorted(items), "common_supported": common,
               "common_untruncated": untruncated,
               "api_compatible_untruncated": [key for key in untruncated if audits[key]["api_valid"]]}
    summary = {"runtime": runtime, "incomplete": incomplete, "cohorts": {}}
    lines = ["# English System One comparison", "",
             "Fixed GLiFormer-selected sample; batch 1, one question per forward, FP16 on the GTX 1060.",
             "Native tokenizers, question templates, and probability heads differ. No thresholds or prompts were tuned on these answers.",
             "GLiFormer noul uses one sigmoid label; candidates use native binary false/true softmax.",
             "All-selected accuracy counts unsupported or failed inputs as incorrect. Compare common-untruncated accuracy alongside coverage.",
             "These public datasets may overlap model training data. Results are not an unseen-data guarantee.", ""]
    if incomplete:
        lines += ["Incomplete candidates: " + ", ".join(incomplete) + ". Comparisons include only complete runs.", ""]
    for cohort, keys in cohorts.items():
        result = {"n": len(keys), "models": {}}
        lines += [f"## {cohort} ({len(keys)} cases)", "", "| Model | Accuracy | Delta vs base (95% CI) |", "|---|---:|---:|"]
        first = [predictions["base"][key] for key in keys]
        for name, rows in predictions.items():
            ordered = [rows[key] for key in keys]
            score = metrics(ordered)
            difference = paired(first, ordered)
            result["models"][name] = {"metrics": score, "paired_vs_base": difference}
            if score["accuracy"] is not None:
                delta = (f"{difference['delta'] * 100:+.2f} pp [{difference['ci95'][0] * 100:+.2f}, {difference['ci95'][1] * 100:+.2f}]"
                         if name != "base" else "reference")
                lines.append(f"| {name} | {score['accuracy'] * 100:.2f}% | {delta} |")
        result["datasets"] = {}
        for dataset in sorted({item["dataset"] for item in items.values()}):
            dataset_keys = [key for key in keys if items[key]["dataset"] == dataset]
            result["datasets"][dataset] = {name: metrics([rows[key] for key in dataset_keys]) for name, rows in predictions.items()}
        result["by_type"] = {kind: {name: metrics([rows[key] for key in keys if items[key]["request"]["questions"]["q"]["type"] == kind])
                                    for name, rows in predictions.items()} for kind in ("choice", "noul")}
        summary["cohorts"][cohort] = result
        lines.append("")
    lines += ["## Runtime", "", "| Model | p50 | p95 | Peak board VRAM |", "|---|---:|---:|---:|"]
    for name, record in runtime.items():
        memory = f"{record['board_peak_mib']:.0f} MiB" if record.get("board_peak_mib") is not None else "not sampled in the earlier run"
        if record["p50_ms"] is not None:
            lines.append(f"| {name} | {record['p50_ms']:.1f} ms | {record['p95_ms']:.1f} ms | {memory} |")
    lines += ["", "Latency includes preparation and decoding, excludes HTTP and model loading. Earlier GLiFormer timing used its existing wrapper.",
              "Board VRAM is sampled every 50 ms, so short spikes can be missed; PyTorch allocation peaks are also in summary.json.",
              "Laya keeps its shipped calibration with the pinned library's temperature clamping. Probabilities are not recalibrated here.", ""]
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "summary.md").write_text("\n".join(lines))
    print(f"Summary: {args.results / 'summary.md'}")


if __name__ == "__main__":
    main()
