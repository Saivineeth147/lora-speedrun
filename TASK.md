# Task Specs

This document is the human-readable rulebook. The machine-readable values live in the
spec files ([`spec.yaml`](./spec.yaml) for Track 1, [`spec-t2.yaml`](./spec-t2.yaml) for
Track 2), which the harness enforces. If they ever disagree, the spec files win.

## Tracks

Records live per track; a frozen track never changes. Tracks are deliberately different
model families and task types, so a technique only proves general by winning on more
than one — single-track wins are real records but weaker evidence of transfer.

| Track | Base model | Task / data | Metric | Status |
|---|---|---|---|---|
| **t1** | Qwen/Qwen2.5-1.5B | GSM8K math (train split only) | ≥ 57.0% exact-match | **frozen** (`spec-v1-frozen`) |
| **t2** | HuggingFaceTB/SmolLM2-1.7B | SQuAD v1.1 extractive QA (train split only) | ≥ 60% EM *(provisional)* | freezing at t2 baseline |

Hardware (1× L40S, Modal network-blocked sandbox), the ≤30M adapter-param cap, the
60-minute run limit, and the 3-fresh-seed verification protocol are identical across
tracks. Submissions declare their track in `config.yaml` (`track: t1` or `track: t2`).

---

# Track 1: `gsm8k-qwen2.5-1.5b-l40s`

## Freeze status

**FROZEN in full as of 2026-07-18** (tag: `spec-v1-frozen`): base model, dataset, eval
protocol, hardware, constraints, verification rules, and the target.

The target (57.0%) was calibrated from the maintainer's baseline run on spec hardware
(58.91% observed with textbook-default LoRA; ~2pts headroom for seed variance) — high
enough that clearing it requires genuinely training the model, low enough that speed
optimizations have room to be aggressive.

## The task

Starting from the frozen base model, produce a PEFT adapter such that the model clears the
target GSM8K accuracy under the fixed evaluation protocol. **Your score is the wall-clock
time of your training run.** Lower is better.

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B` — base, not instruct, revision pinned in `harness/pins.json` |
| Training data | `openai/gsm8k`, config `main`, **train split only** (7,473 examples) |
| Eval | GSM8K test (1,319 problems), protocol below |
| Target | ≥ 57.0% exact-match — frozen |
| Hardware | 1× NVIDIA L40S (48 GB), Modal sandbox, network-blocked |
| Trainable params | ≤ 30,000,000, adapter-only |
| Run limit | Training must finish in ≤ 60 minutes or it won't be verified |

## What is timed

Wall-clock of the training process: from the moment the harness launches
`python train.py ...` to the moment it exits — model loading, tokenization, training,
adapter saving, everything. Model and dataset downloads are **not** timed: the harness
prefetches them into a cache volume (`harness/prefetch.py`) and runs training fully
offline (`HF_HUB_OFFLINE=1` plus a network-blocked sandbox). Sandbox/container startup
is not timed either — the clock starts when your process launches inside the running
sandbox (`harness/modal_verify.py`).

Official times come from the Modal L40S only. Local runs (any GPU) are great for
iteration but are never leaderboard times.

The harness sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for every run to
avoid allocator-fragmentation OOM warnings on sandbox cold starts. It affects memory
management only (not training math), and is identical for all submissions.

Evaluation is not timed (it's identical for everyone), but it is run by the harness
immediately after training, and it is what decides pass/fail.

## The submission contract

Your submission is a directory `submissions/NNN-yourhandle/` containing at minimum
`config.yaml`, `train.py`, and `NOTES.md` (see `submissions/TEMPLATE/`). The harness invokes:

```
python train.py --data-dir <path-to-gsm8k-train> --output-dir <dir> --seed <int>
```

- `--data-dir` points at a `datasets.load_from_disk`-loadable copy of the **train split only**.
- `--seed <int>` is chosen by the verifier at rerun time. Your script must consume it; results
  must not depend on one lucky seed (all 3 verification runs must pass).
- On exit, `<output-dir>` must contain a standard PEFT adapter
  (`adapter_config.json` + `adapter_model.safetensors`) loadable by the pinned `peft`
  version against the frozen base model. The harness evaluates base + your adapter.

## Evaluation protocol (fixed, identical for all submissions)

Implemented in [`harness/evaluate_gsm8k.py`](./harness/evaluate_gsm8k.py). No variations.

- Prompt: `Question: {question}\nAnswer:` — no few-shot examples, no chat template.
- Greedy decoding, bf16, `max_new_tokens=512`, batch size 64, generation truncated at any
  hallucinated `Question:` continuation.
- Answer extraction: last match of `####\s*([-+]?[\d,]*\.?\d+)` in the generation; commas
  stripped; numeric compare vs. the gold answer. No match = wrong. Your model must learn to
  emit the `#### <number>` format — that's part of the task.

## Allowed

- Any LoRA-family / PEFT technique loadable by the pinned `peft` (LoRA, QLoRA, DoRA, rsLoRA,
  PiSSA init, any rank/placement/schedule), within the trainable-param cap.
- Quantizing the frozen base for training (NF4, int8, …). Eval always runs base in bf16 +
  your adapter.
- Any subset, ordering, formatting, packing, or masking of the GSM8K train split.
- Carving your own validation set out of the train split.
- Custom CUDA/Triton kernels, `torch.compile`, Unsloth/Liger-style patches — if the source
  ships in your submission or is an open-source pinned pip package listed in your
  `config.yaml`. Installation happens before timing starts.
- Internal early-stopping logic, as long as it never touches the test split.

## Banned — instant reject, repeat offenses = ban

- Training on, evaluating against, or conditioning any decision on the **GSM8K test split**.
- Any training data beyond the GSM8K train split: no other datasets, no synthetic data,
  no self-generated data, no distillation from any other model.
- Modifying base model weights (the adapter cap is enforced by counting every tensor in
  your saved adapter — `modules_to_save` full-layer copies count against the 30M cap).
- Hardcoding answers, answer lookup structures, or eval-format exploits in the adapter path.
- Tampering with the harness, the cached base model, or the timer.
- Non-reproducible single-seed luck: all 3 fresh-seed verification runs must clear target.

Gray areas: open an issue *before* submitting. Maintainer rulings get added to this file.

## Verification (summary — full protocol in JUDGING.md, security model in SECURITY.md)

1. An automated Claude security screen reviews your diff and posts findings publicly.
2. A maintainer reviews your code, then triggers `/verify`: your `train.py` reruns 3×
   with fresh seeds inside a network-blocked Modal sandbox on the spec L40S, pinned env
   (`env.lock`), pinned artifacts (`harness/pins.json`).
3. All 3 runs must clear target. Official time = mean of the 3 training wall-clocks.
4. Adapter param count audited from the safetensors file; base model and dataset content
   hashes re-verified in a fresh sandbox before every eval (anti-tampering).
5. Public verification report posted on the PR and committed to `records/verifications/`.

---

# Track 2: `squad-smollm2-1.7b-l40s`

Machine-readable spec: [`spec-t2.yaml`](./spec-t2.yaml). Everything not listed below
(what is timed, submission contract, allowed/banned techniques, verification) is
identical to Track 1 — reread those sections with "SQuAD train split" substituted for
"GSM8K train split".

| Item | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-1.7B` — base, ungated, revision pinned in `harness/pins-t2.json` |
| Training data | `rajpurkar/squad` (SQuAD v1.1), **train split only** (87,599 examples — any subset/order/format) |
| Eval | SQuAD v1.1 validation (10,570 questions), fixed protocol below |
| Target | ≥ 60% exact match *(PROVISIONAL — freezes from the t2 baseline runs, same protocol as t1: observed − ~2pts)* |
| Everything else | identical to Track 1 (hardware, caps, timing, verification) |

## Track 2 evaluation protocol (fixed)

Implemented in [`harness/evaluate_squad.py`](./harness/evaluate_squad.py). No variations.

- Prompt: `Context: {context}\nQuestion: {question}\nAnswer:` — 0-shot, no chat template.
- Greedy, bf16, `max_new_tokens=32`, batch 128, generation truncated at hallucinated
  continuations; prediction = first line.
- Score: **exact match against ANY of the gold answers** using the official SQuAD
  normalization (lowercase; strip punctuation, articles, extra whitespace).

## Why this pairing

Deliberate maximum contrast with Track 1: different model family (SmolLM2 vs Qwen —
also sidesteps "Qwen is math-contaminated" concerns), different capability (reading
comprehension vs math reasoning), different data shape (long context + short answer vs
short question + long chain-of-thought). Techniques that transfer across both are
probably real; techniques that don't are overfits to a setup — which is exactly what
the track system exists to expose.
