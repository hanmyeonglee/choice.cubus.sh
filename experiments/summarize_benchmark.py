"""Report paired quality differences and complete-corpus token-limit coverage."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else None


def auc(rows):
    examples = sorted((row["answer"]["noul"], bool(row["gold"])) for row in rows)
    positives = sum(gold for _, gold in examples)
    negatives = len(examples) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0
    index = 0
    while index < len(examples):
        end = index + 1
        while end < len(examples) and examples[end][0] == examples[index][0]:
            end += 1
        rank_sum += (index + 1 + end) / 2 * sum(gold for _, gold in examples[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(rows):
    valid = [row for row in rows if row["status"] == "ok"]
    bins = defaultdict(list)
    for row in valid:
        bins[min(int(row["confidence"] * 10), 9)].append(row)
    ece = sum(len(group) * abs(statistics.mean(row["confidence"] for row in group) -
                               statistics.mean(row["correct"] for row in group)) for group in bins.values()) / len(valid) if valid else None
    labels = set(row["gold"] for row in valid) | set(row["prediction"] for row in valid)
    f1s = []
    for label in labels:
        true_positive = sum(row["gold"] == row["prediction"] == label for row in valid)
        false_positive = sum(row["prediction"] == label and row["gold"] != label for row in valid)
        false_negative = sum(row["gold"] == label and row["prediction"] != label for row in valid)
        denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(2 * true_positive / denominator if denominator else 0)
    gold_counts = Counter(row["gold"] for row in rows)
    return {"n": len(rows), "errors": len(rows) - len(valid),
            "gold_counts": {str(key): value for key, value in gold_counts.items()},
            "majority_accuracy": max(gold_counts.values()) / len(rows) if rows else None,
            "accuracy": sum(row.get("correct", False) for row in rows) / len(rows) if rows else None,
            "macro_f1": statistics.mean(f1s) if f1s else None,
            "brier": statistics.mean(row["brier"] for row in valid) if valid else None,
            "ece": ece, "auc": auc(valid) if valid and all(row["answer"]["type"] == "noul" for row in valid) else None}


def paired(base, large, iterations=2000):
    grouped = defaultdict(list)
    wins = losses = 0
    for first, second in zip(base, large):
        if first["id"] != second["id"]:
            raise ValueError("Paired rows do not match.")
        difference = int(second.get("correct", False)) - int(first.get("correct", False))
        wins += difference == 1
        losses += difference == -1
        grouped[(first["dataset"], str(first["group"]))].append(difference)
    clusters = [(sum(values), len(values)) for values in grouped.values()]
    if not clusters:
        return None
    rng = random.Random(42)
    draws = []
    for _ in range(iterations):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draws.append(sum(total for total, _ in sampled) / sum(count for _, count in sampled))
    return {"delta": (wins - losses) / len(base), "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
            "large_only_correct": wins, "base_only_correct": losses, "clusters": len(clusters)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--base", default="base-predictions.jsonl")
    parser.add_argument("--large", default="large-predictions.jsonl")
    args = parser.parse_args()
    runtime = {name: json.loads((args.results / filename).with_suffix(".meta.json").read_text())
               for name, filename in (("base", args.base), ("large", args.large))}
    base = {row["id"]: row for row in load(args.results / args.base)}
    large = {row["id"]: row for row in load(args.results / args.large)}
    selection = load(args.results / "selection.jsonl")
    expected_ids = {row["id"] for row in selection}
    if set(base) != expected_ids or set(large) != expected_ids:
        raise ValueError("Both evaluation runs must be complete before reporting final quality.")
    audit = {row["id"]: row for row in load(args.results / "base-lengths.jsonl")}
    other_audit = {row["id"]: row for row in load(args.results / "large-lengths.jsonl")}
    grouped = defaultdict(list)
    for row in audit.values():
        grouped[row["dataset"]].append(row)
    quality = []
    for dataset, length_rows in sorted(grouped.items()):
        ids = sorted(row["id"] for row in selection if row["dataset"] == dataset)
        first = [base[key] for key in ids]
        second = [large[key] for key in ids]
        lengths = [max(row["tokens"], other_audit[row["id"]]["tokens"]) for row in length_rows]
        quality.append({"dataset": dataset, "corpus_n": len(length_rows),
                        "over_1024": sum(length > 1024 for length in lengths),
                        "over_1024_rate": sum(length > 1024 for length in lengths) / len(lengths),
                        "tokens_p50": percentile(lengths, 0.5), "tokens_p95": percentile(lengths, 0.95), "tokens_max": max(lengths),
                        "api_valid_n": sum(row["api_valid"] for row in length_rows),
                        "base": metrics(first), "large": metrics(second), "paired": paired(first, second)})
    ids = sorted(expected_ids)
    api_ids = [key for key in ids if audit[key]["api_valid"]]
    overall = {"base": metrics([base[key] for key in ids]), "large": metrics([large[key] for key in ids]),
               "paired": paired([base[key] for key in ids], [large[key] for key in ids]),
               "macro_dataset_accuracy_base": statistics.mean(row["base"]["accuracy"] for row in quality),
               "macro_dataset_accuracy_large": statistics.mean(row["large"]["accuracy"] for row in quality),
               "api_compatible": {"base": metrics([base[key] for key in api_ids]), "large": metrics([large[key] for key in api_ids]),
                                  "paired": paired([base[key] for key in api_ids], [large[key] for key in api_ids])}}
    overall["by_type"] = {}
    for kind in ("choice", "noul"):
        typed_ids = sorted(row["id"] for row in selection if row["request"]["questions"]["q"]["type"] == kind)
        typed_datasets = defaultdict(list)
        for key in typed_ids:
            typed_datasets[base[key]["dataset"]].append(key)
        dataset_rows = [{"dataset": dataset, "base": metrics([base[key] for key in keys]),
                         "large": metrics([large[key] for key in keys])}
                        for dataset, keys in sorted(typed_datasets.items())]
        overall["by_type"][kind] = {
            "base": metrics([base[key] for key in typed_ids]), "large": metrics([large[key] for key in typed_ids]),
            "paired": paired([base[key] for key in typed_ids], [large[key] for key in typed_ids]),
            "macro_accuracy_base": statistics.mean(row["base"]["accuracy"] for row in dataset_rows),
            "macro_accuracy_large": statistics.mean(row["large"]["accuracy"] for row in dataset_rows),
            "datasets": dataset_rows,
        }
        if kind == "noul":
            overall["by_type"][kind]["macro_auc_base"] = statistics.mean(row["base"]["auc"] for row in dataset_rows if row["base"]["auc"] is not None)
            overall["by_type"][kind]["macro_auc_large"] = statistics.mean(row["large"]["auc"] for row in dataset_rows if row["large"]["auc"] is not None)
    summary = {"runtime": runtime, "quality": quality, "overall": overall,
               "lengths": {"total": len(audit), "over_1024": sum(row["over_1024"] for row in audit.values()),
                           "api_valid": sum(row["api_valid"] for row in audit.values()),
                           "api_valid_over_1024": sum(row["api_valid"] and row["over_1024"] for row in audit.values()),
                           "different_checkpoint_lengths": sum(row["tokens"] != other_audit[key]["tokens"] for key, row in audit.items())}}
    if all(metadata["device"] == "cuda" and metadata["batch_size"] == 1 for metadata in runtime.values()):
        summary["latency_ms"] = {}
        for name, predictions in (("base", base), ("large", large)):
            times = [row["seconds"] for row in predictions.values()]
            summary["latency_ms"][name] = {"p50_ms": 1000 * statistics.median(times),
                                          "p95_ms": 1000 * percentile(times, 0.95),
                                          "sum_seconds": sum(times)}
    (args.results / "quality-summary.json").write_text(json.dumps(summary, indent=2))
    environment = f"{runtime['base']['device']} {runtime['base']['dtype']}; identical production decision mapping; shared <=1,024-token sample."
    lines = ["# Public choice/noul benchmark", "", environment, "",
             "| Dataset | Corpus n | >1024 tokens | Evaluated n | Base accuracy | Large accuracy | Delta pp (95% paired cluster CI) |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in quality:
        comparison = row["paired"]
        delta = f"{100 * comparison['delta']:+.1f} [{100 * comparison['ci95'][0]:+.1f}, {100 * comparison['ci95'][1]:+.1f}]" if comparison else "n/a"
        lines.append(f"| {row['dataset']} | {row['corpus_n']} | {row['over_1024']} ({100 * row['over_1024_rate']:.1f}%) | {row['base']['n']} | {100 * row['base']['accuracy']:.1f}% | {100 * row['large']['accuracy']:.1f}% | {delta} |")
    lines.extend(["", "Noul discrimination and raw score calibration:", "",
                  "| Dataset | Base AUROC | Large AUROC | Base Brier | Large Brier |", "|---|---:|---:|---:|---:|"])
    for row in quality:
        if row["base"]["auc"] is not None and row["large"]["auc"] is not None:
            lines.append(f"| {row['dataset']} | {row['base']['auc']:.4f} | {row['large']['auc']:.4f} | {row['base']['brier']:.4f} | {row['large']['brier']:.4f} |")
    lines.extend(["", "Overall:", "", "```json", json.dumps(overall, indent=2), "```", "", "Length coverage:", "", "```json", json.dumps(summary["lengths"], indent=2), "```"])
    report = "\n".join(lines) + "\n"
    (args.results / "quality-tables.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
