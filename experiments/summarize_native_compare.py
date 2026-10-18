"""Report native-interface quality on identical cases, with coverage and paired intervals."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import statistics


MODELS = ("base", "large", "gliner-decide", "laya")


def clean(row):
    return row["status"] == "ok" and not any(row["audit"].get(key, False)
                                            for key in ("over_limit", "state_truncated", "head_truncated"))


def random_choice_p_value(rows, correct):
    """Exact upper-tail probability under independent uniform candidate guessing."""
    distribution = [1.0]
    for row in rows:
        chance = 1 / row["candidate_count"]
        updated = [0.0] * (len(distribution) + 1)
        for hits, probability in enumerate(distribution):
            updated[hits] += probability * (1 - chance)
            updated[hits + 1] += probability * chance
        distribution = updated
    return min(1.0, sum(distribution[correct:]))


def metrics(rows):
    if not rows:
        return {"n": 0, "correct": 0, "accuracy": None, "macro_f1": None, "chance": None}
    labels = sorted({row["gold"] for row in rows} | {row["prediction"] for row in rows})
    f1s = []
    for label in labels:
        tp = sum(row["gold"] == label and row["prediction"] == label for row in rows)
        fp = sum(row["gold"] != label and row["prediction"] == label for row in rows)
        fn = sum(row["gold"] == label and row["prediction"] != label for row in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    correct = sum(row["correct"] for row in rows)
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    # Exam answer strings differ by question; a global class F1 has no task meaning.
    fixed_labels = not any(row["dataset"] == "mmlu_pro" for row in rows)
    macro_f1 = statistics.mean(f1s) if fixed_labels else None
    majority = max(Counter(row["gold"] for row in rows).values()) / len(rows) if fixed_labels else None
    chance = statistics.mean(1 / row["candidate_count"] for row in rows)
    if len(by_dataset) > 1 and fixed_labels:
        per_dataset = [metrics(group) for group in by_dataset.values()]
        macro_f1 = statistics.mean(value["macro_f1"] for value in per_dataset)
        majority = statistics.mean(value["majority_baseline"] for value in per_dataset)
        chance = statistics.mean(value["chance"] for value in per_dataset)
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "macro_f1": macro_f1, "majority_baseline": majority,
            "random_choice_upper_tail_p": random_choice_p_value(rows, correct) if not fixed_labels else None,
            "dataset_macro_accuracy": statistics.mean(statistics.mean(r["correct"] for r in group)
                                                        for group in by_dataset.values()),
            "chance": chance}


def compare(first, second, ids):
    if not ids:
        return None
    groups = defaultdict(list)
    for key in ids:
        groups[first[key]["dataset"]].append(int(first[key]["correct"]) - int(second[key]["correct"]))
    rng = random.Random(42)
    # Equal dataset weight matches the operational-domain macro metric.
    draws = sorted(statistics.mean(statistics.mean(rng.choices(values, k=len(values)))
                                   for values in groups.values()) for _ in range(2000))
    return {"n": len(ids), "dataset_macro_delta": statistics.mean(statistics.mean(v) for v in groups.values()),
            "ci95": [draws[50], draws[1949]], "datasets": len(groups),
            "first_only_correct": sum(first[key]["correct"] and not second[key]["correct"] for key in ids),
            "second_only_correct": sum(second[key]["correct"] and not first[key]["correct"] for key in ids)}


def fmt(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    items = [json.loads(line) for line in (args.results / "items.jsonl").read_text().splitlines()]
    ids = {row["id"] for row in items}
    predictions = {}
    metadata = {}
    coverage = {}
    for model in MODELS:
        path = args.results / f"{model}.jsonl"
        metadata[model] = json.loads(path.with_suffix(".meta.json").read_text())
        if not metadata[model].get("complete") or metadata[model].get("smoke"):
            raise ValueError(f"Incomplete full evaluation: {model}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if model == "laya":
            extra = args.results / "laya-published-boolq.jsonl"
            extra_meta = json.loads(extra.with_suffix(".meta.json").read_text())
            if not extra_meta.get("complete") or extra_meta["items_sha256"] != metadata[model]["items_sha256"]:
                raise ValueError("The published Laya BoolQ template evaluation is incomplete or uses different inputs.")
            extra_rows = [json.loads(line) for line in extra.read_text().splitlines()]
            expected = {r["id"] for r in rows if r["dataset"] == "boolq"}
            if len(extra_rows) != len(expected) or {r["id"] for r in extra_rows} != expected:
                raise ValueError("The published Laya BoolQ rows must cover identical reading cases.")
            metadata[model]["published_boolq"] = extra_meta
            rows = [row for row in rows if row["dataset"] != "boolq"] + extra_rows
        predictions[model] = {row["id"]: row for row in rows}
        if len(rows) != len(items) or set(predictions[model]) != ids:
            raise ValueError(f"Mismatched or duplicated case IDs: {model}")
        successful = [r for r in rows if r["status"] == "ok"]
        coverage[model] = {"requested": len(rows), "status": dict(Counter(r["status"] for r in rows)),
                           "untruncated": sum(clean(r) for r in rows),
                           "state_truncated": sum(r["audit"].get("state_truncated", False) for r in rows),
                           "head_truncated": sum(r["audit"].get("head_truncated", False) for r in rows),
                           "correct_over_all_requested": sum(r["correct"] for r in successful) / len(rows)}
    if len({m["items_sha256"] for m in metadata.values()}) != 1:
        raise ValueError("The models did not evaluate identical source inputs.")
    common = sorted(key for key in ids if all(clean(predictions[m][key]) for m in MODELS))
    report = {"metadata": metadata, "coverage": coverage, "common_untruncated_n": len(common),
              "suites": {}, "datasets": {}, "computer_science": {}, "pairwise": {},
              "native_all_answered": {}}
    for model in MODELS:
        report["native_all_answered"][model] = {}
        for suite in ("operations", "knowledge", "reading"):
            rows = [row for row in predictions[model].values() if row["suite"] == suite]
            ok = [row for row in rows if row["status"] == "ok"]
            report["native_all_answered"][model][suite] = {
                "requested": len(rows), "answered": len(ok), "untruncated": sum(clean(r) for r in rows),
                "answered_only": metrics(ok),
                "correct_over_requested": sum(r["correct"] for r in ok) / len(rows),
            }
    lines = ["# Native-interface comparison", "",
             "Four fixed checkpoints, public SDK prediction methods, CPU FP32. No prompt tuning or result reuse.",
             "Operational inputs are the public development split, with one single-label head per document. Published held-out scores are not being reproduced.",
             "Laya reading results use its author's published BoolQ dictionary-state/noul template; BoolQ was in its training mix. The generic-choice diagnostic remains in laya.jsonl but is not used as its primary reading score.",
             "The primary comparison uses identical cases that fit every model without state or head truncation. Context coverage is reported separately.",
             "No combined intelligence ranking is inferred from an arbitrary mixture of operational, knowledge, and reading tasks.", "",
             "## Coverage", "", "| Model | Requested | Successful | Untruncated | Unsupported | State cut | Head cut |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model, cov in coverage.items():
        lines.append(f"| {model} | {cov['requested']} | {cov['status'].get('ok', 0)} | {cov['untruncated']} | {cov['status'].get('unsupported', 0)} | {cov['state_truncated']} | {cov['head_truncated']} |")
    lines += ["", f"Common untruncated cases: {len(common)}.", "", "## Primary suite results", "",
              "| Suite | Cases | Chance | Base | Large | GLiNER Decide | Laya |", "|---|---:|---:|---:|---:|---:|---:|"]
    for suite in ("operations", "knowledge", "reading"):
        selected = [key for key in common if predictions['base'][key]['suite'] == suite]
        values = {m: metrics([predictions[m][key] for key in selected]) for m in MODELS}
        report["suites"][suite] = values
        lines.append(f"| {suite} | {len(selected)} | {fmt(values['base']['chance'])} | " + " | ".join(fmt(values[m]['dataset_macro_accuracy'] if selected else None) for m in MODELS) + " |")
        report["pairwise"][suite] = {f"{a}_minus_{b}": compare(predictions[a], predictions[b], selected)
                                     for a, b in combinations(MODELS, 2)}
    lines += ["", "Operational scores above give equal weight to each represented domain. Other suites each contain one dataset.", "",
              "## Per-dataset results on the common untruncated subset", "",
              "| Dataset | Cases | Base | Large | GLiNER Decide | Laya |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in sorted({r["dataset"] for r in items}):
        selected = [key for key in common if predictions['base'][key]['dataset'] == dataset]
        values = {m: metrics([predictions[m][key] for key in selected]) for m in MODELS}
        report["datasets"][dataset] = values
        lines.append(f"| {dataset} | {len(selected)} | " + " | ".join(fmt(values[m]['accuracy']) for m in MODELS) + " |")
    selected = [key for key in common if predictions['base'][key]['category'] == 'computer science']
    report["computer_science"] = {m: metrics([predictions[m][key] for key in selected]) for m in MODELS}
    report["pairwise"]["computer_science"] = {
        f"{a}_minus_{b}": compare(predictions[a], predictions[b], selected)
        for a, b in combinations(MODELS, 2)
    }
    lines += ["", f"## Computer science ({len(selected)} common cases)", ""]
    for m in MODELS:
        value = report['computer_science'][m]
        lines.append(f"- {m}: {value['correct']}/{value['n']} ({fmt(value['accuracy'])})")
    lines += ["", "## Knowledge versus uniform random choice", "",
              "Exact one-sided Poisson-binomial upper-tail test, using each question's own candidate count. This is an exploratory random-guessing null, not a measure of general intelligence.", "",
              "| Model | Correct | Cases | Chance | Random-choice upper-tail p |", "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        value = report["suites"]["knowledge"][model]
        p_value = value.get("random_choice_upper_tail_p")
        lines.append(f"| {model} | {value['correct']} | {value['n']} | {fmt(value['chance'])} | {p_value:.6g} |" if p_value is not None else f"| {model} | 0 | 0 | n/a | n/a |")
    lines += ["", "## Paired model differences", "",
              "Stratified paired bootstrap over cases within datasets; 2,000 draws, seed 42, exploratory 95% intervals without multiple-comparison adjustment.", "",
              "| Suite | First minus second | Cases | Difference | 95% interval |", "|---|---|---:|---:|---:|"]
    for suite, pairs in report['pairwise'].items():
        for pair, value in pairs.items():
            if value:
                lines.append(f"| {suite} | {pair} | {value['n']} | {value['dataset_macro_delta']*100:+.1f} pp | [{value['ci95'][0]*100:+.1f}, {value['ci95'][1]*100:+.1f}] |")
    lines += ["", "## Native results over all requested cases", "",
              "These secondary figures include native truncation. Unsupported inputs count as incorrect in the final column. Answered-only denominators differ and must not be compared as paired scores.", "",
              "| Model | Suite | Requested | Answered | Untruncated | Answered-only accuracy | Correct / requested |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for model, suites in report["native_all_answered"].items():
        for suite, values in suites.items():
            lines.append(f"| {model} | {suite} | {values['requested']} | {values['answered']} | {values['untruncated']} | {fmt(values['answered_only']['accuracy'])} | {fmt(values['correct_over_requested'])} |")
    (args.results / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.results / "summary.md").write_text("\n".join(lines) + "\n")
    (args.results / "common-untruncated-ids.json").write_text(json.dumps(common, indent=2) + "\n")
    print(args.results / "summary.md")


if __name__ == "__main__":
    main()
