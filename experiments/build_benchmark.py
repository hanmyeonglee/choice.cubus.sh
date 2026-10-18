"""Normalize downloaded public choice/noul cases without changing their text."""

import argparse
import json
from pathlib import Path


def clean_question(question):
    # The production noul API accepts only the statement, with no criteria.
    keys = ("type", "instructions", "criteria") if question["type"] == "choice" else ("type", "instructions")
    return {key: question[key] for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "datasets")
    args = parser.parse_args()
    items = []
    omitted = []
    for path in sorted((args.data / "jevbench").glob("*.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            if row["question"]["type"] not in ("choice", "noul"):
                omitted.append({"suite": "jevbench", "id": row["id"], "reason": "unsupported_type"})
                continue
            gold = row["expected"]
            if gold is None:
                omitted.append({"suite": "jevbench", "id": row["id"], "reason": "no_gold"})
                continue
            if row["question"]["type"] == "noul":
                if gold not in ("yes", "no"):
                    raise ValueError(f"Unexpected noul gold: {gold!r}")
                gold = gold == "yes"
            items.append({"id": "jevbench/" + row["id"], "dataset": "jevbench/" + path.stem,
                          "group": row.get("group") or row["id"], "family": row["family"], "gold": gold,
                          "request": {"state": row["state"], "questions": {"q": clean_question(row["question"])}}})
    for path in sorted((args.data / "jeff").glob("*.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            items.append({"id": "jeff/" + row["id"], "dataset": "jeff/" + path.stem,
                          "group": row["id"], "gold": row["gold"],
                          "request": {"state": row["state"], "questions": {"q": clean_question(row["questions"]["q"])}}})
    for path in sorted((args.data / "leanmcp").glob("*.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            if row["perturbation"] != "none":
                continue
            items.append({"id": "leanmcp/" + row["case_id"], "dataset": "leanmcp/" + path.stem,
                          "group": row["family_id"], "gold": row["gold"],
                          "license": row["license"], "source_revision": row["revision"],
                          "source_truncated": row.get("truncated", False),
                          "request": {"state": row["state"], "questions": {"q": clean_question(row["question"])}}})
    if len({row["id"] for row in items}) != len(items):
        raise ValueError("Duplicate evaluation IDs.")
    (args.data / "items.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in items))
    (args.data / "omitted.json").write_text(json.dumps(omitted, indent=2))
    print(json.dumps({"cases": len(items), "omitted": len(omitted)}))


if __name__ == "__main__":
    main()
