# English System One candidates

This isolated experiment compares the existing GLiFormer base/large FP16 results with two English models on the GTX 1060 3 GB. It does not change the choice API or its configured model.

| Candidate | Actual checkpoint parameters | FP16 parameter storage only | Native inference |
|---|---:|---:|---|
| `fastino/GLiNER2.5-Decide` | 486,444,053 | 928 MiB | DeBERTa encoder and classification head |
| `convaiinnovations/laya` | 421,293,827 | 804 MiB | ModernBERT encoder and typed decision head |

These storage figures exclude CUDA context, activations, buffers, and allocator reserves. They are not measured total VRAM. The GLiNER model card advertises 340M, but the pinned checkpoint contains 486,444,053 parameters including embeddings and heads.

Both projects use Apache 2.0. All downloads are public and require no Hugging Face account. Model revisions, file hashes, inspected library source revisions, and source archive hashes are pinned in `system_one_models.json`. The existing PyTorch 2.7.1+cu126 runtime is retained for Pascal support. FlashAttention, TileLang, compilation, BF16, quantization, and CPU fallback are not enabled in this experiment.

## Protocol

- Reuse the exact 3,345 selected cases and existing GLiFormer FP16 predictions from `quality-gpu-20261018T030324Z`.
- One question per forward, batch size 1, model parameters converted to FP16 on CUDA. CPU compatibility checks use FP32.
- Serialize states exactly as the existing API does. Preserve instructions, option order, labels, and descriptions.
- GLiNER uses the library's schema classification path, question instructions as the schema prompt, and the official runtime's native scoring function over all choice labels. FP16 softmax is preserved; probabilities are not promoted to FP32 before activation. Noul becomes a native two-label `false: No` / `true: Yes` question, retaining both native probabilities for verification. This differs from the existing GLiFormer wrapper's single-label sigmoid noul mapping.
- GLiNER prepared subword sequences above 1,024 tokens are marked unsupported; options and states are not silently shortened. Its `max_len` library argument counts words, so it is deliberately not used as a subword-token guard.
- Laya retains its checkpoint defaults: 512 total tokens, 192 question/option tokens, and shipped temperatures with the pinned library's clamping. Record state truncation, instruction/option truncation, and label distinguishability. Do not silently raise its context budget or tune its templates.
- Before evaluating, verify both adapters against their respective public APIs on one choice and one noul case.
- Report all-selected accuracy, supported coverage, and matched supported/untruncated cohorts. Report API-compatible matched cases separately. A larger tokenizer can exceed its budget on a case GLiFormer handled; do not discard that limitation from the coverage report.
- Compare accuracy with paired cluster bootstrap confidence intervals, dataset and question-type breakdowns, noul AUROC/Brier/ECE, p50/p95 latency, and memory peaks.
- Sample board VRAM every 50 ms and also record PyTorch's peak allocated/reserved bytes. Board sampling can miss brief spikes. No other workload should use the GPU during measurement.
- Warm up before timing. Latency includes adapter preprocessing and decoding, excludes HTTP and loading. Existing GLiFormer timings come from its earlier wrapper run, so small timing differences should not be overinterpreted.

Public benchmark data may overlap training data. These scores measure this reproducible public suite and do not establish unseen-task quality. The current suite is English; it does not evaluate Korean or multilingual capability.

## Run on driftwood

The prepared bundle is `/home/nemo/choice-system-one-probe`. Its `experiments` directory contains the scripts, fixed sample, paired baselines, model weights, and inspected library sources. Its sibling `app` contains only the existing schema helpers.

```bash
sudo python3 /home/nemo/choice-system-one-probe/experiments/run_system_one_gpu.py \
  --compose-dir /home/nemo/cubus.sh
```

The runner verifies/downloads assets, resolves the available service image by name or ID, gives it a unique experiment tag, and builds a separate small dependency layer. If neither recorded reference is available locally, it builds a private base image from `sites/choice/Dockerfile`. It keeps named references for both build and run operations. These steps do not recreate the service container or replace its image tag. CPU preflight checks verify that PyTorch matches the paired baseline while the API is available. It then stops the original choice container, checks 20 task families on CUDA, and runs each candidate over the fixed sample. A `finally` block restarts the same original container after ordinary errors, interrupts, or completion. An abrupt host shutdown or SIGKILL cannot execute that cleanup; restart the original choice container manually in that situation.

Results are written to `experiments/results/system-one-gpu-<timestamp>/`, including `summary.md`, `summary.json`, full predictions, runtime metadata, GPU samples, and error logs. Runtime duration is not yet measured on the 1060; the runner permits up to 30 minutes per candidate invocation. It does not rebuild or replace the production service.

Every invocation evaluates both candidates from the first case and writes a new result directory. Candidate predictions from previous runs are never imported or resumed.

## Other candidates

- NanoJev (`C-Tianyu/NanoJev`) uses a Qwen3-0.6B backbone, but the current official release is trained for Maze, Snake, ViZDoom, and Predict Position. It is a possible research comparison, with a different task specialization, rather than the first general English routing candidate.
- Current Kev 0.8B advertises a 4 GB GPU target. It is not assumed to fit a 3 GB card without measuring its custom runtime. The older Kev 0.5B is explicitly marked superseded.

Primary sources: [GLiNER model](https://huggingface.co/fastino/GLiNER2.5-Decide), [GLiNER source](https://github.com/fastino-ai/GLiNER2), [Laya model](https://huggingface.co/convaiinnovations/laya), [Laya source](https://github.com/NandhaKishorM/laya), [NanoJev model](https://huggingface.co/C-Tianyu/NanoJev), [Kev source](https://github.com/jaredpalmer/kev).
