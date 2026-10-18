# Public choice/noul quality benchmark

## Scope

The primary completed evaluation compares the current production mapping for
`knowledgator/gliformer-base-v1` (264,196,614 parameters) and
`knowledgator/gliformer-large-v1` (575,637,510 parameters). It uses CUDA FP16,
eager attention, batch size 1, and four Torch threads on GTX 1060 3 GB.
These are model quality and inference measurements; timing excludes HTTP and
network overhead. Production remains unchanged. NF4 accuracy is not measured.

Both checkpoints completed every one of the same 3,345 selected cases with
zero failures. The full length audits were performed on CPU FP32. Local CPU
FP32 inference started as a reference and was stopped after the complete GPU
results arrived; those partial predictions are diagnostic records, not the
primary benchmark scores. Metadata identifies device and precision.

Sources are pinned by revision and file SHA-256 in `benchmark_sources.json`:

- [Benchmark Heaven JevBench](https://github.com/fstandhartinger/jevbench):
  public easy, original, and hard cases. The private portions are unavailable.
- [Jeff evaluation samples](https://github.com/logan-markewich/jeff/tree/main/bench):
  AG News, emotion, BoolQ, irony, SMS spam, and SST-2; 200 cases each.
  Jeff already shortened long source passages to about 2,000 characters.
- [Leanmcp JevBench](https://huggingface.co/datasets/Leanmcp/jevbench):
  eleven public text slices; canonical `perturbation=none` rows only.
  Reordered copies are excluded to avoid counting the same case twice.

This yields 6,429 choice/noul cases across 20 dataset slices. Score questions
and image inputs are excluded. Source text, instructions, choice order, and
choice descriptions are retained. Noul `criteria` are removed because the
production API supports only its `type` and `instructions`; the current model
mapping does not consume noul criteria. State objects use the production
compact JSON serializer. No new text truncation, prompt optimization,
calibration, or test-set threshold tuning is applied. Each upstream source
keeps its own license; cached data and generated reports are excluded from Git.

## Length audit and selection

Both models audit every case with their own installed processor and collator.
The main encoder's complete `attention_mask` length includes schema markers,
instructions, labels, descriptions, and source text. During the audit,
`config.max_len` is temporarily raised to prevent native truncation from
hiding over-limit cases. No encoder forward pass is performed during counting.
The native checkpoint budget is recorded separately.

All 6,429 measured lengths match between the two checkpoints. The common
quality sample includes only cases at or below 1,024 tokens for both models:
uniform sampling without replacement, up to 200 per slice, using seed 42
and a stable per-slice SHA-256-derived seed. The same 3,345 selected cases
are used for both models. This is a conditional quality comparison on shorter
inputs, not an evaluation of model quality on the excluded long cases.

Current API field validation is audited independently. Model-only quality
evaluation also includes cases that violate other existing API limits, such
as 77 choices or long instructions. Results for the API-compatible selected
subset are reported separately. Coverage rates describe these downloaded
public slices, not the entire upstream datasets or a production traffic mix.

## Metrics

- Accuracy uses the top choice, or raw noul score >= 0.5.
- Macro F1 uses the observed union of gold and predicted labels per slice.
- Noul AUROC measures separation independently of the 0.5 threshold.
- Brier uses the multiclass sum; noul uses `2 * (score - target)^2`.
- ECE uses ten equal-width bins of top-label confidence.
- Accuracy differences use 2,000 paired cluster-bootstrap draws, seed 42,
  resampling scenario groups to account for related JevBench paraphrases.
- Errors count as incorrect for accuracy. Native official composite scores,
  private tiers, hosted Jev results, and cost scores are not reproduced.

The partial CPU reference starts at batch size 1 and resumes at size 4 to reduce CPU time.
Before using batches, one case from every slice is compared against the
single-request mapping. Both the top decisions and score tolerance are
checked; per-run maximum differences are stored in metadata. These are
ordinary padded model batches with separate class groups for each item.
The primary GPU run uses batch size 1 throughout.

## Reproduce

Download public inputs without an account, then normalize them:

```bash
python3 experiments/fetch_benchmark.py
python3 experiments/build_benchmark.py
```

Run `benchmark.py` in the production image's dependency environment, with
the repository's `app/` available on `PYTHONPATH`. The scripts import it but
never start the HTTP server or require API keys.

```bash
python experiments/benchmark.py audit --checkpoint /path/to/base \
  --items experiments/datasets/items.jsonl --output results/base-lengths.jsonl
python experiments/benchmark.py audit --checkpoint /path/to/large \
  --items experiments/datasets/items.jsonl --output results/large-lengths.jsonl
python3 experiments/select_benchmark.py results

python experiments/benchmark.py evaluate --checkpoint /path/to/base \
  --items results/selection.jsonl --output results/base-predictions.jsonl \
  --batch-size 4 --verify-batch
python experiments/benchmark.py evaluate --checkpoint /path/to/large \
  --items results/selection.jsonl --output results/large-predictions.jsonl \
  --batch-size 4 --verify-batch
python3 experiments/summarize_benchmark.py results
```

Prediction files append one record per case and resume from completed IDs.
Metadata records checkpoint, parameter count, precision, threads, batch size,
previous invocations, and batch verification. Keep inputs and selection fixed
when resuming. Hashes of the checkpoint files and scripts should be retained
with the report for an archived run.

## Optional server FP16 rerun

Copy `benchmark.py`, `precision.py`, `run_quality_gpu.py`, the application's
`app/` directory, and the fixed `selection.jsonl` to the standalone experiment
directory. Place the selection at `experiments/datasets/quality-selection.jsonl`.
The large checkpoint must already exist under `experiments/models/large/`.

```bash
sudo python3 experiments/run_quality_gpu.py --compose-dir /path/to/cubus.sh
```

This pauses only the running `choice` service and gives each model up to
900 seconds, then restarts the original container. No production setting is
changed. As with the precision runner, SIGKILL or host/daemon failure cannot
run cleanup. Copy the completed prediction files back beside the original
length audits and selection before running the same summary script.
