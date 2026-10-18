# Native-interface comparison

This study compares GLiFormer base, GLiFormer large, GLiNER2.5-Decide, and the English Laya checkpoint through their official public SDK prediction methods. It does not modify deployment, tune prompts on gold answers, or reuse previous prediction files.

## Fixed inputs

- Operational decisions: all 1,700 public development documents from `fastino/fast-decisions`, pinned at `35afb79a0f611a1db44380e855ac2a821b0a4da1`. Select the first single-label head per document, in source order. This gives 100 documents per domain and avoids converting multi-label tasks into single-choice questions. This differs from scoring all heads and is not a reproduction of the unpublished 5,100-document test split.
- Knowledge: all 500 already-pinned MMLU-Pro cases. Use complete answer texts as candidates and report computer science separately. Questions, choices, and answers were previously joined to the pinned original test data.
- Reading: all 200 already-pinned Jeff BoolQ examples. Preserve passage and question, with yes/no candidates.

All models receive identical source information and candidate meanings. The interface mapping is fixed before prediction; no per-task winning prompt is selected from earlier ablations. SDK input syntax necessarily differs between models. An official SDK interface does not imply an officially published benchmark adapter for every dataset.

## Prediction methods

- GLiFormer: `classify`, `multi_label=False`, `threshold=1e-6`. Bare candidate list for exams and BoolQ; native named task group for operational decisions so labels such as yes/no retain the task meaning. BoolQ includes passage and question in the source prose.
- GLiNER: `classify_text` or `batch_classify_text`, original task name, label list, `multi_label=False`, and the question prompt for question-answering tasks. This matches the developer's published operational and passage-question API patterns.
- Laya operational/exam tasks: `predict` or `predict_batch`, raw text state and typed `choice`, with the original question or readable task name and bare candidate labels. Retain the shipped state and head budgets. No automatic model router, tournament, or fine-tuning is used.
- Laya BoolQ: the exact published `en.boolq` template from `research/scripts/build_benchmark_nb.py`: dictionary state with passage and question, and the noul instruction `Based on \`passage\`, is the answer to \`question\` yes?`. Run this separately with `native_laya_boolq.py` and use it as Laya's primary reading result. The generic-choice diagnostic is retained separately. The author explicitly identifies BoolQ as a training task, so its reading score measures retention, not demonstrated out-of-task generalization. This source-based follow-up was added before any new BoolQ predictions were available.

Public batch predictions are checked against the public single-item call for the first supported case per dataset. Input auditing may use SDK internals; predictions do not use custom model forwards or decoders.

## Execution and reporting

Run offline on local CPU FP32, four Torch threads per model, public batches of up to four states sharing a schema. The existing local GPU has little free memory; production is not interrupted. This measures quality, not GTX 1060 GPU latency or deployment feasibility.

Count complete prepared GLiFormer/GLiNER lengths and skip inputs above 1,024 tokens. Preserve Laya's native truncation and record state/head loss. Compare accuracy on the exact common untruncated cases, with coverage reported separately. Operational scores use equal domain weight; knowledge and reading are separate suites. Report per-dataset scores, chance baselines, macro-F1, and paired bootstrap intervals. No aggregate mixture is presented as a universal intelligence score.

Source manifests, input hashes, prediction files, metadata, and script snapshots accompany each result directory. Use a new output directory for every run.

## Commands

Download the pinned public development data with its checked hashes:

```bash
python experiments/fetch_benchmark.py --manifest experiments/native_benchmark_sources.json
```

The existing normalized inputs and original subject join are prepared by the
earlier quality benchmark pipeline described in `BENCHMARK.md`.

Inside the prepared benchmark runtime:

```bash
python experiments/native_compare.py prepare --data experiments/datasets --output experiments/results/NEW_RUN
python experiments/native_compare.py evaluate --model base --checkpoint /opt/models/gliformer-base-v1 --items experiments/results/NEW_RUN/items.jsonl --output experiments/results/NEW_RUN/base.jsonl
```

Repeat evaluation with `large`, `gliner-decide`, and `laya` and their pinned local checkpoints. Then:

```bash
python experiments/summarize_native_compare.py experiments/results/NEW_RUN
```

The GLiNER and Laya source checkouts and additional runtime dependencies must be on `PYTHONPATH`, as in the existing System One image. Historical Jev-shaped experiments use `legacy_schemas.py`, independently of the service API being edited.

Sources: [fast-decisions](https://huggingface.co/datasets/fastino/fast-decisions), [GLiFormer evaluator](https://github.com/Knowledgator/GLiFormer/blob/b7d11a995a24d8e39ad141ecc0beae449faf6441/gliformer_eval/eval_classification.py), [GLiNER model card](https://huggingface.co/fastino/GLiNER2.5-Decide), [Laya SDK](https://github.com/NandhaKishorM/laya/tree/d74e8be46ef5241e4ba2b16cd25e896f24f71373).
