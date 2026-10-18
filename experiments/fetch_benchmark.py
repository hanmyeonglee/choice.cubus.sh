"""Download public evaluation inputs at pinned revisions and verify SHA-256."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "datasets")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("benchmark_sources.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())

    def fetch(source):
        path = args.data / source["local_path"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]:
            return source["local_path"]
        request = urllib.request.Request(source["url"], headers={"User-Agent": "choice-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {source['local_path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return source["local_path"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for path in pool.map(fetch, manifest["files"]):
            print(path)
    if manifest.get("metadata_path"):
        path = args.data / manifest["metadata_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
