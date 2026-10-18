"""Prepare two candidate runtimes, benchmark on the GPU, and restore choice."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import uuid


def command(*argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, **kwargs)


def output(*argv):
    return command(*argv, stdout=subprocess.PIPE).stdout.strip()


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}.")


def prepare_base(container, tag, compose_dir):
    references = output("docker", "container", "inspect", "--format",
                        "{{.Image}}\n{{.Config.Image}}", container).splitlines()
    recorded_id, configured_reference = references
    # A container's recorded ID may no longer resolve in the image store.
    # Use an available named reference, and keep using our unique tag afterward.
    for reference in dict.fromkeys((configured_reference, recorded_id)):
        inspected = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if inspected.returncode == 0:
            command("docker", "image", "tag", reference, tag)
            print(f"Using local runtime base: {reference}", flush=True)
            return {"container_image_id": recorded_id, "configured_reference": configured_reference,
                    "resolved_reference": reference, "resolved_image_id": inspected.stdout.strip(),
                    "rebuilt": False}
    # Rebuild a private experiment image if both original references are gone.
    # This neither recreates the service container nor changes its image tag.
    context = compose_dir.resolve() / "sites/choice"
    dockerfile = context / "Dockerfile"
    if not dockerfile.is_file():
        raise RuntimeError("Neither service image reference exists locally, and the service Dockerfile is unavailable.")
    print("Service image references are unavailable; building a private runtime base from sites/choice.", flush=True)
    command("docker", "build", "-t", tag, "-f", str(dockerfile), str(context))
    return {"container_image_id": recorded_id, "configured_reference": configured_reference,
            "resolved_reference": tag,
            "resolved_image_id": output("docker", "image", "inspect", "--format", "{{.Id}}", tag),
            "rebuilt": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-dir", type=Path, required=True)
    args = parser.parse_args()
    probe = Path(__file__).resolve().parent
    sample = probe / "datasets/quality-selection.jsonl"
    baseline = probe / "baseline"
    expected_torch = json.loads((baseline / "base-predictions.meta.json").read_text())["torch"]
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    ids = {row["id"] for row in rows}
    if len(rows) != 3345 or len(ids) != 3345:
        raise ValueError("The fixed 3,345-case quality sample is required.")
    for model in ("base", "large"):
        predictions = [json.loads(line) for line in (baseline / f"{model}-predictions.jsonl").read_text().splitlines()]
        if len(predictions) != len(rows) or {row["id"] for row in predictions} != ids:
            raise ValueError("Complete paired GLiFormer baseline predictions are required.")
    container = output("docker", "compose", "--project-directory", str(args.compose_dir.resolve()), "ps", "-q", "choice")
    if not container or "\n" in container:
        raise RuntimeError("Exactly one running choice container is required.")
    identifier = uuid.uuid4().hex[:10]
    base_tag = f"choice-system-one-base:{identifier}"
    image = f"choice-system-one-probe:{identifier}"
    name = f"choice-system-one-{identifier}"
    directory = probe / "results" / ("system-one-gpu-" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True)
    provenance = {
        "production_container": container,
        "items_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "scripts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in probe.glob("*system*.*") if path.is_file()},
        "gpu_before": output("nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv"),
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Preparing models and runtime while choice is running. Results: {directory}", flush=True)
    command(sys.executable, str(probe / "fetch_system_one.py"))
    try:
        provenance["runtime_base"] = prepare_base(container, base_tag, args.compose_dir)
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
        command("docker", "build", "--build-arg", f"BASE_IMAGE={base_tag}", "-t", image,
                "-f", str(probe / "SystemOne.Dockerfile"), str(probe))
    finally:
        subprocess.run(["docker", "image", "rm", base_tag],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    runtime_image = output("docker", "image", "inspect", "--format", "{{.Id}}", image)
    (directory / "runtime-image.txt").write_text(runtime_image + "\n")

    def run(model, device, limit=None, suffix=""):
        result = f"{model}{suffix}-predictions.jsonl"
        argv = ["docker", "run", "--rm", "--name", name, "--network", "none", "--user", "0:0",
                "--memory", "10g", "--memory-swap", "10g", "--entrypoint", "python",
                "--mount", f"type=bind,source={probe},target=/probe,readonly",
                "--mount", f"type=bind,source={probe.parent / 'app'},target=/app/app,readonly",
                "--mount", f"type=bind,source={directory},target=/results"]
        if device == "cuda":
            argv += ["--gpus", "all"]
        argv += [image, "/probe/system_one.py", "--model", model,
                 "--checkpoint", f"/probe/models/system-one/{model}",
                 "--items", "/probe/datasets/quality-selection.jsonl", "--output", f"/results/{result}",
                 "--device", device, "--precision", "fp16" if device == "cuda" else "fp32", "--verify"]
        if limit:
            argv += ["--limit", str(limit)]
        with (directory / f"{model}{suffix}.log").open("w") as log:
            try:
                command(*argv, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            except subprocess.SubprocessError as error:
                log.flush()
                details = "\n".join((directory / f"{model}{suffix}.log").read_text().splitlines()[-40:])
                raise RuntimeError(f"Failed {model}{suffix}. Last log lines:\n{details}") from error
            finally:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        predictions = [json.loads(line) for line in (directory / result).read_text().splitlines()]
        metadata = json.loads((directory / result).with_suffix(".meta.json").read_text())
        if metadata["torch"] != expected_torch:
            raise RuntimeError(f"Runtime PyTorch {metadata['torch']} differs from the paired baseline {expected_torch}.")
        if len(predictions) != (limit or 3345) or any(row["status"] == "error" for row in predictions):
            raise RuntimeError(f"Incomplete or failed {model} run. Inspect {directory / (model + suffix + '.log')}.")

    for model in ("gliner-decide", "laya"):
        run(model, "cpu", limit=2, suffix="-preflight")
    signal.signal(signal.SIGTERM, interrupt)
    paused = False
    failures = []
    try:
        paused = True
        command("docker", "stop", "--timeout", "30", container)
        for model in ("gliner-decide", "laya"):
            print(f"Running {model} FP16 on the GPU.", flush=True)
            with (directory / f"{model}-gpu.csv").open("w") as monitoring:
                sampler = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits", "-lms", "50"], stdout=monitoring, stderr=subprocess.DEVNULL)
                try:
                    run(model, "cuda", limit=20, suffix="-smoke")
                    run(model, "cuda")
                except (subprocess.SubprocessError, RuntimeError) as error:
                    failures.append({"model": model, "error": str(error)})
                    print(f"Failed {model}: {error}", flush=True)
                finally:
                    sampler.terminate()
                    sampler.wait(timeout=10)
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if paused:
            command("docker", "start", container)
            print("Restarted the original choice container.", flush=True)
    (directory / "failures.json").write_text(json.dumps(failures, indent=2))
    command(sys.executable, str(probe / "summarize_system_one.py"), str(directory),
            "--baseline", str(baseline), "--items", str(sample))
    print(f"Results: {directory}", flush=True)
    if failures:
        raise SystemExit("At least one candidate failed; inspect failures.json and the per-model logs.")


if __name__ == "__main__":
    main()
