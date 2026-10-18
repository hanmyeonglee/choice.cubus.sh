"""Print per-trial memory, latency, failures, and differences from base FP32."""

import argparse
import csv
import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", type=Path, help="Optional base_fp32.json from an earlier run")
    args = parser.parse_args()
    reports = {}
    for path in args.results.glob("*.json"):
        report = json.loads(path.read_text())
        if isinstance(report, dict) and "cases" in report:
            reports[report["name"]] = report
    baseline_report = json.loads(args.baseline.read_text()) if args.baseline else reports.get("base_fp32", {})
    baseline = {case["name"]: case for case in baseline_report.get("cases", []) if case["status"] == "ok"}
    gpu_peaks = {}
    outcomes_path = args.results / "outcomes.json"
    gpu_path = args.results / "gpu.csv"
    if outcomes_path.exists() and gpu_path.exists():
        host_path = args.results / "host.json"
        offset = json.loads(host_path.read_text())["utc_offset_seconds"] if host_path.exists() else 0
        timezone = datetime.timezone(datetime.timedelta(seconds=offset))
        samples = []
        with gpu_path.open() as source:
            for raw_row in csv.DictReader(source):
                row = {key.strip(): value.strip() for key, value in raw_row.items()}
                timestamp = datetime.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone).timestamp()
                samples.append((timestamp, int(row["memory.used [MiB]"])))
        for trial in json.loads(outcomes_path.read_text()):
            values = [used for timestamp, used in samples if trial["started_unix"] <= timestamp <= trial["ended_unix"]]
            gpu_peaks[trial["name"]] = max(values) if values else "n/a"
    print("| Trial | Finished | OK / cases | Sampled device peak MiB | Peak allocated / reserved MiB | Inference seconds | Choice changes / comparisons | Max noul delta |")
    print("|---|---|---|---|---|---|---|---|")
    for name, report in sorted(reports.items()):
        cases = report["cases"]
        successful = [case for case in cases if case["status"] == "ok"]
        memories = [report.get("after_load", {}), report.get("failure_memory", {}), report.get("final_memory", {})]
        for case in cases:
            memories.extend([case.get("memory", {}), case.get("gpu_failure", {}).get("memory", {})])
        allocated = max((item.get("peak_allocated_mib", 0) for item in memories), default=0)
        reserved = max((item.get("peak_reserved_mib", 0) for item in memories), default=0)
        changed = comparisons = 0
        noul_deltas = []
        for case in successful:
            original = baseline.get(case["name"], {}).get("answers", {})
            for key, answer in case["answers"].items():
                reference = original.get(key)
                if reference is None:
                    continue
                if answer["type"] == "choice":
                    comparisons += 1
                    changed += answer["choice"] != reference["choice"]
                else:
                    noul_deltas.append(abs(answer["noul"] - reference["noul"]))
        seconds = sum(case.get("total_with_fallback_seconds", case["seconds"]) for case in cases)
        delta = f"{max(noul_deltas):.5f}" if noul_deltas else "n/a"
        print(f"| {name} | {report['completed']} | {len(successful)} / {len(cases)} | {gpu_peaks.get(name, 'n/a')} | {allocated:.0f} / {reserved:.0f} | {seconds:.2f} | {changed} / {comparisons} | {delta} |")
    for name, report in sorted(reports.items()):
        if report.get("error"):
            print(f"\n{name}: {report['error']}\n")
    if outcomes_path.exists():
        for trial in json.loads(outcomes_path.read_text()):
            if trial.get("timed_out") or trial.get("exit_code", 0):
                print(f"\nRunner failure: {trial}")


if __name__ == "__main__":
    main()
