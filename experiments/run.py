"""Run isolated GPU trials, then restart the original API container."""

import argparse
import datetime
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


def command(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def output(*args):
    return command(*args, stdout=subprocess.PIPE).stdout.strip()


def stop_on_signal(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent.parent / "app")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--case-filter", default="")
    args = parser.parse_args()
    probe = Path(__file__).resolve().parent
    source = args.source_dir.resolve()
    if not (source / "model.py").is_file():
        parser.error("--source-dir must point to the application's app directory.")
    matrix = json.loads((args.matrix or probe / "matrix.json").read_text())
    if not matrix:
        parser.error("The experiment matrix must not be empty.")
    for trial in matrix:
        if not re.fullmatch(r"[a-z0-9_]+", trial["name"]):
            parser.error("Trial names must contain only lowercase letters, digits, and underscores.")
        for key, allowed, default in (
            ("model", ("base", "large"), "base"),
            ("precision", ("fp32", "fp16"), "fp32"),
            ("quantization", ("none", "nf4"), "none"),
            ("memory_mode", ("gpu", "cpu-fallback"), "gpu"),
        ):
            if trial.get(key, default) not in allowed:
                parser.error(f"Invalid {key} in {trial['name']}.")
        if not 1 <= trial.get("timeout_seconds", 180) <= 1800:
            parser.error("Trial timeouts must be between 1 and 1800 seconds.")
        if trial.get("allocator", "") not in ("", "expandable_segments:True"):
            parser.error("Unsupported allocator experiment.")
    needs_large = any(trial["model"] == "large" for trial in matrix)
    existing_weights = sum(path.stat().st_size for pattern in ("*.safetensors", "*.bin")
                           for path in (probe / "models" / "large").glob(pattern))
    if needs_large and shutil.disk_usage(probe).free < max(0, 2_303_000_000 - existing_weights) + 1024**3:
        raise RuntimeError("Insufficient free disk space for the large checkpoint and experiment image.")
    container = output("docker", "compose", "--project-directory", str(args.compose_dir.resolve()), "ps", "-q", "choice")
    if not container or "\n" in container:
        raise RuntimeError("Exactly one running choice container is required.")
    image = output("docker", "inspect", "--format", "{{.Image}}", container)
    token = uuid.uuid4().hex[:10]
    base_tag = f"choice-probe-base:{token}"
    probe_tag = f"choice-precision-probe:{token}"
    trial_container = f"choice-probe-{token}"
    results = probe / "results" / datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    results.mkdir(parents=True)
    (results / "matrix.json").write_text(json.dumps(matrix, indent=2))
    (results / "host.json").write_text(json.dumps({
        "utc_offset_seconds": datetime.datetime.now().astimezone().utcoffset().total_seconds(),
        "base_image": image,
    }, indent=2))
    common_mounts = [
        "--mount", f"type=bind,source={probe},target=/probe,readonly",
        "--mount", f"type=bind,source={source},target=/app/app,readonly",
    ]
    stopped = False
    monitor = None
    monitor_file = None
    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        # Setup finishes while the API is still running.
        command("docker", "tag", image, base_tag)
        command("docker", "build", "--build-arg", f"BASE_IMAGE={base_tag}", "-t", probe_tag, str(probe))
        if needs_large:
            models = probe / "models"
            models.mkdir(exist_ok=True)
            download_code = (
                "from huggingface_hub import snapshot_download; "
                "snapshot_download('knowledgator/gliformer-large-v1', "
                "revision='d0a4e53d09cebe6bc963dd9be319d4279084bb2d', "
                "local_dir='/models/large', allow_patterns="
                "['*.json', '*.safetensors', '*.bin', '*.model', '*.txt'])"
            )
            command("docker", "run", "--rm", "--name", trial_container, "--user", "0:0",
                    "--entrypoint", "python", "-e", "HF_HUB_OFFLINE=0", "-e", "TRANSFORMERS_OFFLINE=0",
                    "--mount", f"type=bind,source={models},target=/models", probe_tag, "-c", download_code)
            weights = probe / "models" / "large" / "pytorch_model.bin"
            if not weights.is_file() or weights.stat().st_size != 2_302_735_855:
                raise RuntimeError("The pinned large checkpoint weights are missing or incomplete; the API has not been stopped.")
            print("Checking large checkpoint loading on CPU before pausing the API.", flush=True)
            with (results / "large_preflight.log").open("w") as log:
                command("docker", "run", "--rm", "--name", trial_container,
                        "--network", "none", "--memory", "10g", "--memory-swap", "10g",
                        "--entrypoint", "python", *common_mounts, probe_tag, "-c",
                        "from legacy_schemas import load_gliformer as load_model; model=load_model('/probe/models/large','cpu'); "
                        "print('Loaded parameters:',sum(p.numel() for p in model.parameters()))",
                        stdout=log, stderr=subprocess.STDOUT)
        print(f"Setup complete. Pausing choice; results: {results}", flush=True)
        # Set this before stopping so cleanup also covers an interrupted stop.
        stopped = True
        command("docker", "stop", "--time", "30", container)
        monitor_file = (results / "gpu.csv").open("w")
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu",
            "--format=csv,nounits", "-lms", "20",
        ], stdout=monitor_file, stderr=subprocess.STDOUT)
        outcomes = []
        for trial in matrix:
            name = trial["name"]
            checkpoint = "/opt/models/gliformer-base-v1" if trial["model"] == "base" else "/probe/models/large"
            argv = [
                "docker", "run", "--rm", "--name", trial_container, "--gpus", "all",
                "--network", "none", "--user", "0:0", "--memory", "10g", "--memory-swap", "10g",
                "-e", f"PYTORCH_CUDA_ALLOC_CONF={trial.get('allocator', '')}",
                "--entrypoint", "python", *common_mounts,
                "--mount", f"type=bind,source={results},target=/results", probe_tag,
                "/probe/precision.py", "--name", name, "--checkpoint", checkpoint,
                "--precision", trial.get("precision", "fp32"),
                "--quantization", trial.get("quantization", "none"),
                "--memory-mode", trial.get("memory_mode", "gpu"),
                "--case-filter", args.case_filter or trial.get("case_filter", ""), "--output", f"/results/{name}.json",
            ]
            print(f"Running {name}", flush=True)
            started = time.time()
            outcome = {"name": name, "started_unix": started}
            with (results / f"{name}.log").open("w") as log:
                try:
                    completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=trial.get("timeout_seconds", 180))
                    outcome["exit_code"] = completed.returncode
                except subprocess.TimeoutExpired:
                    outcome["timed_out"] = True
                finally:
                    subprocess.run(["docker", "rm", "-f", trial_container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            outcome["ended_unix"] = time.time()
            outcomes.append(outcome)
            (results / "outcomes.json").write_text(json.dumps(outcomes, indent=2))
    finally:
        # Cleanup is restricted to containers and tags created by this run.
        subprocess.run(["docker", "rm", "-f", trial_container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if monitor_file is not None:
            monitor_file.close()
        if stopped:
            command("docker", "start", container)
            print("Restarted the original choice container.", flush=True)
        subprocess.run(["docker", "image", "rm", probe_tag, base_tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Results: {results}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
