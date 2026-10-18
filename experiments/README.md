# Isolated precision and memory experiments

For the comparison through public native model interfaces, see
[NATIVE_COMPARISON.md](NATIVE_COMPARISON.md). Historical Jev-shaped evaluations
use the frozen `legacy_schemas.py` definitions rather than the current API schema.

For the public dataset quality comparison and input-token coverage audit,
see [BENCHMARK.md](BENCHMARK.md).

These tools do not change the production model or API. They extend the running
service image with bitsandbytes 0.50.2 in a separate image and run synthetic
inputs without API keys or network access during inference.

## Variants

| Variant | What changes |
|---|---|
| Base / large FP32 | Full precision reference and larger checkpoint |
| Base / large FP16 | Floating point parameters use half precision |
| Base / large NF4 | Only encoder Linear weights use 4-bit NF4; compute remains FP32 |
| Base CPU fallback | On GPU OOM, release the GPU model and reload FP32 on CPU |

FP16 is a floating point precision change, rather than integer quantization.
It reduces floating point weight storage, but does not halve every temporary
tensor. The relative-position integer tensors that caused the observed OOM
are unaffected. Pascal GPUs can also execute FP16 arithmetic slowly.

NF4 leaves embeddings and task heads in FP32. It is not a fully 4-bit model.
The retry matrix additionally tests NF4 with FP16 embeddings, heads, and compute.
Only NF4 is tested: bitsandbytes lists Pascal support for NF4/FP4, while
LLM.int8 requires newer GPUs. Successful inference and acceptable output
quality still need to be established for this custom model.

CPU fallback reruns the failed request and stays on CPU for the rest of that
trial. It does not implement CUDA managed memory, automatic RAM pooling, or
layer offloading. CPU reload and retry times are recorded separately.

## Run on the server

Requirements: Python 3.11 or later, Docker access, NVIDIA Container Toolkit,
`nvidia-smi`, the running Compose service `choice`, and outbound access during
image setup and the optional model download. No Hugging Face login is needed.

From this repository:

```bash
sudo python3 experiments/run.py --compose-dir /path/to/cubus.sh
```

For a standalone copy of this directory, also copy the application's `app/`
directory and pass its path with `--source-dir`.

1. The runner builds the experimental image and downloads the pinned public
   large checkpoint while the API is running. The large weights need about
   2.3 GB of disk space; setup also needs space for the experimental image.
2. It stops only the existing `choice` container for exclusive use of its GPU.
   **The API is unavailable during the trials.** The default matrix allows up
   to 23 minutes of inference time, plus container and model loading overhead.
3. Each trial uses a fresh container, a 180-second timeout (300 for CPU fallback),
   and a 10 GiB host memory cap with swap disabled for that container.
4. Cleanup restarts the original container on completion, ordinary errors,
   timeouts, SIGINT, or SIGTERM. SIGKILL, power loss, or daemon failure cannot
   run cleanup; recover with `docker compose start choice` if needed.

Use `--matrix /path/to/matrix.json` to select variants, or
`--case-filter billing_prose` for a short smoke run. Copy entries from
`matrix.json` into a smaller matrix to avoid large-model downloads and shorten
the API pause. Check the JSON reports and logs even when the container exits
successfully: model errors are recorded as data.

After the initial run, use `--matrix experiments/retry-matrix.json` to rerun
the large models, test FP16 with expandable allocator segments, combine NF4
and FP16, and give the Unicode CPU fallback up to 900 seconds. The retry
matrix allows up to 33 minutes of trials. Expandable segments change allocator
behavior and do not provide RAM fallback. The large checkpoint is downloaded
including its `.bin` weights and loaded on CPU before the API is stopped.

## Results

```bash
python3 experiments/summarize.py experiments/results/<timestamp>
```

Reports include startup memory, PyTorch peak allocated/reserved VRAM, host peak
RSS, per-case times and scores, and explicit OOM failures. The host `gpu.csv`
samples total device use every 20 ms, including driver overhead. Use both GPU
sampling and PyTorch peaks: sampling can miss brief peaks, and reserved memory
is retained allocator space rather than live tensors alone.

Inputs comprise ten functional examples and three requests at all API field
limits: readable prose, dense punctuation, and Unicode symbols. All requests
are checked against the existing API schema. One pass is an initial screen;
it is not a throughput benchmark or proof that every valid input will fit.
The few expected labels and changes from FP32 are diagnostic checks, not a
representative quality evaluation. The noul 0.5 threshold is illustrative.

The large checkpoint is pinned to
`knowledgator/gliformer-large-v1@d0a4e53d09cebe6bc963dd9be319d4279084bb2d`.
Generated models, reports, and logs are excluded from Git and production builds.

## Decision representation quality

[FORMAT_ABLATION.md](FORMAT_ABLATION.md) documents the controlled CPU FP32
comparison of current requests, semantic class labels, short task names,
native classification calls, and state serialization. These experiments
hold examples and checkpoints fixed and do not change the running API.
`summarize_format_ablation.py` reports all conditions with paired differences
and input-truncation audits.

References:

- [bitsandbytes hardware requirements](https://huggingface.co/docs/bitsandbytes/main/en/installation)
- [NVIDIA Pascal mixed precision characteristics](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/)
- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-memory-management)
