# Decision representation experiment

The experiment asks whether the current Jev-shaped wrapper changes accuracy relative to each model's native classification interface. No API, model deployment, weights, or hard limits are changed.

## Fixed controls

- Use the same 200 SST-2, 200 AG News, 200 emotion, and 15 MMLU-Pro computer science examples already present in the fixed quality sample.
- Keep gold answers, option meaning, model checkpoints, precision, and device fixed within each comparison. Run all baselines again on local CPU FP32, batch 4, four threads. The earlier GPU FP16 figures are context, not the control for these new comparisons.
- Change one representation component at a time where possible. The native reference changes a documented bundle of interface conventions; its gain cannot be attributed to one component alone.
- No training or probability calibration. All variants are reported rather than selecting a winning prompt on the evaluation answers.
- Verify paired batch versus single-item decisions on the first two cases of each condition. Verify native calls against the public library method. Record true prepared input lengths and Laya truncation flags.
- Class-name and description changes are lexical/task-encoding interventions. They are not merely changes to JSON whitespace.

## Main matrix

| Condition | Change |
|---|---|
| `current` | Exact current request and model-specific decision mapping |
| `binary_choice` | SST-2 only: replace noul with two choice options, `false: No` and `true: Yes`; keep the question |
| `bare_labels` | SST-2: `negative`/`positive` options with the original question. News/emotion: retain class names and remove descriptions. CS: use the complete answer text as the class label and remove opaque letter IDs |
| `short_task` | Keep `bare_labels` and replace the question with `sentiment`, `topic`, `emotion`, or `answer` |
| `native` | GLiFormer: bare class list, single-label mode, threshold 1e-6, no named question group. GLiNER: short task name, bare class list, no extra prompt. Laya: typed choice still requires an instruction, so this matches its `short_task` representation |
| `plain_question` | CS only: unwrap `{"question": "..."}` into the question text, retaining the current question and options |

The main matrix has 2,675 model-condition cases per model and 615 unique source cases. Class predictions are mapped back to the original gold space before scoring. The native AG News vocabulary and order follow the official GLiFormer evaluator's first-seen SetFit test labels: `Business`, `Sci/Tech`, `Sports`, `World`.

The existing GLiNER adapter replaces an absent description with the label itself before building its dictionary schema. Consequently `bare_labels` means no explicit description in the request, but GLiNER still receives a label-as-description fallback. Its `native` condition passes a true bare list with no descriptions. GLiFormer and Laya render description-free criteria as the label alone. This is an adapter-format difference retained deliberately in the current-versus-native comparison.

## Additional Laya reference checks

The inspected Laya `benchmarks/bench_fast.py` uses a JSON state `{"text": text}`, task-specific category instructions, and its own AG News criteria wording. Therefore the additional Laya-only run compares:

- `json_state`: add only the `text` JSON wrapper to current SST-2/news/emotion requests (600 cases).
- `laya_reference`: reproduce that published benchmark script's request template on the same news/emotion examples (400 cases), including the native dictionary-state serialization with its default JSON spaces, retaining the experiment's CPU FP32 settings. Verify the result against an actual dictionary-state `agent.predict` call.
- `semantic_noul`: an exploratory follow-up on the same 200 SST-2 cases, retaining the noul type, original question, and raw text but providing native true/false criteria `positive sentiment` and `negative sentiment`. The production API currently rejects noul criteria; this is a direct-model diagnostic, not a deployed API change.

These are request-template comparisons on our fixed examples, not reproductions of the published sample or hardware timing. The follow-up conditions are based on source inspection, without choosing templates using the evaluation gold answers.

## Differences from reported benchmarks

GLiFormer's official evaluator passes raw text plus the actual dataset class names, calls `multi_label=False`, and chooses the top class. The current wrapper passes a named question group for choice, concatenates option IDs and descriptions, and turns noul into the raw sigmoid of one question string. Normalizing fixed sigmoid scores alone cannot change their argmax; changes to encoded labels, groups, and binary mappings must be separated from probability presentation.

GLiNER's model-card examples use task schemas such as `sentiment: [negative, positive]`. Our existing wrapper instead encodes a generic `q` task, the question as a prompt, and yes/no labels for noul. Laya supports typed requests natively, but its noul and choice primitives and label wording are separate model inputs. Its published reference script also wraps source text in JSON.

## Reproduction and provenance

Run `format_ablation.py` once per model and `format_laya_reference.py` for Laya's template checks. Use an unused output path for every invocation. `summarize_format_ablation.py <results-directory>` requires all complete outputs and checks identical IDs across conditions. The results include per-case predictions, truncation audits, source hashes, verification metadata, and paired bootstrap intervals.

The official GLiFormer evaluator is pinned at commit `b7d11a995a24d8e39ad141ecc0beae449faf6441`. The Laya reference script comes from the already-pinned SDK commit `d74e8be46ef5241e4ba2b16cd25e896f24f71373`. Model revisions are unchanged from the earlier quality experiments.

Sources: [GLiFormer evaluator](https://github.com/Knowledgator/GLiFormer/blob/b7d11a995a24d8e39ad141ecc0beae449faf6441/gliformer_eval/eval_classification.py), [GLiNER model card](https://huggingface.co/fastino/GLiNER2.5-Decide), [Laya benchmark script](https://github.com/NandhaKishorM/laya/blob/d74e8be46ef5241e4ba2b16cd25e896f24f71373/benchmarks/bench_fast.py).
