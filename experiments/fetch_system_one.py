"""Download pinned public candidate weights and inspected inference sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(url, destination, expected):
    if destination.is_file() and digest(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "choice-system-one-benchmark"})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
        while chunk := source.read(4 * 1024 * 1024):
            target.write(chunk)
    if digest(temporary) != expected:
        temporary.unlink()
        raise ValueError(f"Download hash mismatch: {destination}")
    temporary.replace(destination)
    print(f"Downloaded {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "models/system-one")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("system_one_models.json").read_text())
    jobs = []
    for name, model in manifest["models"].items():
        for entry in model["files"]:
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{entry['path']}"
            jobs.append((url, args.root / name / entry["path"], entry["sha256"]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job: download(*job), jobs))
    for name, source in manifest["sources"].items():
        archive = args.root / f"{name}.tar.gz"
        download(source["url"], archive, source["sha256"])
        destination = args.root / "sources" / name
        if not destination.is_dir():
            with tarfile.open(archive) as bundle:
                bundle.extractall(args.root / "sources", filter="data")
                extracted = args.root / "sources" / bundle.getmembers()[0].name.split("/")[0]
            extracted.rename(destination)
    print(f"Ready: {args.root}", flush=True)


if __name__ == "__main__":
    main()
