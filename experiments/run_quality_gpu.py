"""Compare base and large FP16 on the shared quality sample, restoring the API."""

import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import uuid


def command(*argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, **kwargs)


def output(*argv):
    return command(*argv, stdout=subprocess.PIPE).stdout.strip()


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-dir", type=Path, required=True)
    args = parser.parse_args()
    probe = Path(__file__).resolve().parent
    sample = probe / "datasets/quality-selection.jsonl"
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    if len(rows) != 3345:
        raise ValueError("The fixed 3,345-case quality sample is required.")
    if not (probe / "models/large/pytorch_model.bin").is_file():
        raise ValueError("The previously downloaded large checkpoint is required.")
    container = output("docker", "compose", "--project-directory", str(args.compose_dir.resolve()), "ps", "-q", "choice")
    if not container or "\n" in container:
        raise RuntimeError("Exactly one running choice container is required.")
    image = output("docker", "inspect", "--format", "{{.Image}}", container)
    directory = probe / "results" / ("quality-gpu-" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True)
    name = "choice-quality-" + uuid.uuid4().hex[:10]
    signal.signal(signal.SIGTERM, interrupt)
    paused = False
    try:
        paused = True
        command("docker", "stop", "--time", "30", container)
        for model in ("base", "large"):
            checkpoint = "/opt/models/gliformer-base-v1" if model == "base" else "/probe/models/large"
            with (directory / f"{model}.log").open("w") as log:
                argv = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--network", "none",
                        "--user", "0:0", "--memory", "10g", "--memory-swap", "10g", "--entrypoint", "python",
                        "-e", "PYTHONPATH=/app", "--mount", f"type=bind,source={probe},target=/probe,readonly",
                        "--mount", f"type=bind,source={probe.parent / 'app'},target=/app/app,readonly",
                        "--mount", f"type=bind,source={directory},target=/results", image,
                        "/probe/benchmark.py", "evaluate", "--checkpoint", checkpoint,
                        "--items", "/probe/datasets/quality-selection.jsonl", "--output", f"/results/{model}-predictions.jsonl",
                        "--device", "cuda", "--precision", "fp16", "--batch-size", "1"]
                try:
                    print(f"Running {model} FP16; results: {directory}", flush=True)
                    command(*argv, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                finally:
                    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if paused:
            command("docker", "start", container)
            print("Restarted the original choice container.", flush=True)
    print(f"Results: {directory}", flush=True)


if __name__ == "__main__":
    main()
