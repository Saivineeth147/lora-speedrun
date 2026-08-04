# Verification Report — 005-slippylolo

- **Date:** 2026-08-03
- **Verifier:** Saivineeth147
- **PR:** #14
- **Hardware:** Modal Sandbox · NVIDIA L40S (network-blocked)
- **Harness commit:** (orchestrator checkout) · Python 3.13.12
- **Target:** ≥ 57.0% GSM8K exact-match (spec v1)
- **Seeds:** fresh, generated at verification time (below)

## Timed runs

| run | seed | train wall-clock | GSM8K acc | margin over target | pass |
|-----|------|------------------|-----------|--------------------|------|
| 1 | 1865039302 | 22.0s | 0.579226686884003 | +0.92 pp | ✅ |
| 2 | 43733957 | 21.2s | 0.5905989385898408 | +2.06 pp | ✅ |
| 3 | 163976044 | 21.5s | 0.5708870356330553 | +0.09 pp | ✅ |

**Official train time (all-pass, mean of runs): 21.6s** ·
runs passed: 3/3

## Adapter audit

- Trainable params in saved adapter: **13,762,560** (cap 30,000,000) —
  within cap ✅ — identical across all three seeds
  (6 targets × 28 layers, r=16, `down_proj` dropped).
- Loads in pinned peft against frozen base: ✅ — all three eval runs loaded the saved
  adapter with pinned `peft` against the frozen base and scored, so despite the
  training-time fused/packed representation the export is a standard two-file PEFT
  LoRA covering all 28 layers.
- Integrity re-check: ✅ 3/3 runs reported
  `INTEGRITY: OK (base model + train/test data match committed pins)`.

## Code review notes

- **Training data sources verified train-split-only:** ✅ `train.py` reads only the
  harness-supplied `--data-dir` (via `glob("data-*.arrow")`, with a `load_from_disk`
  fallback) and never opens another data path. No reference to the GSM8K test split
  exists anywhere in the submission.
- **Base-model resolution is the strictest of any submission so far:** `resolve_model_path()`
  builds the snapshot path from the pinned `BASE_MODEL_REVISION`, raises if that exact
  snapshot is absent, has no network fallback, *and* re-reads `harness/pins.json` to
  confirm `base_model` / `base_model_sha` match the candidate before training. This is
  the correct pattern — and notably the opposite of the `sorted(snapshots)[-1]` resolver
  that was rejected from PR #13.
- **No network, no subprocess, no dynamic execution.** A full scan for `requests`/`urllib`/
  `socket`/`http`/`subprocess`/`os.system`/`eval`/`exec`/`__import__`/`pickle`/`base64`
  returns nothing. The only writes are into `--output-dir`; `harness/pins.json` is opened
  read-only. The one file read outside the data dir is the page-warmer, which reads the
  pinned `model.safetensors` and asserts its exact byte size.
- **Technique summary:** Builds on record #4. The pinned Qwen weights are loaded directly
  from safetensors into a meta-constructed model (skipping high-level model/tokenizer
  discovery) while a background sequential read warms the model file during Python
  imports. Training is capped at **12 optimizer updates**: 8 full-network backward passes,
  then a one-time batched materialization of the frozen lower-27-layer boundary
  activations (cached in BF16), followed by 4 updates that backprop through only layer 27,
  the final RMSNorm, and the completion loss. The harness log confirms the transition —
  active parameters drop from 13,271,040 to 491,520 at update 8 while the saved adapter
  stays at 13,762,560. Retiring 27 layers of backward for the tail is where the remaining
  time goes; the full adapter is still active in every forward pass and is exported intact.
- **Concerns:**
  1. **Thin accuracy margin — the main caveat on this record.** Run 3 cleared the bar by
     **0.09 pp**, i.e. about **1.2 problems out of 1,319**. The schedule is tuned close
     enough to the 57.0% target that a marginally unlucky seed could fail it. This does not
     affect eligibility — the rule is 3/3 fresh seeds clearing target, and 3/3 cleared —
     but it is a materially narrower safety margin than any previous record (#4's worst
     seed cleared by 1.9 pp), and it is the reason the anti-resampling rule exists. This
     verification was run **once**, on the submitted code state, and the result stands as
     measured; it was not re-rolled for a better set.
  2. `config.yaml` declares `requested_epoch_fraction: 0.50`, but the 12-update cap means
     the effective epoch fraction is **0.211** (harness log:
     `"effective_epoch_fraction":0.2114..., "capped":true`). The declared field is the
     request, not the realized schedule; noting it so the leaderboard's technique
     description reflects what actually ran.
  3. The author discloses that the build they measured also emitted local profiling and
     reproducibility digests that were stripped from the submitted payload — so the
     submitted source is not byte-identical to what produced their self-reported numbers.
     This is not disqualifying and is in fact the normal case: self-reported numbers are
     advisory, and the official time comes from the maintainer re-running the *submitted*
     code, which is what happened here. Worth noting that the official 21.6s came out
     **faster** than the self-reported 26.83s, not slower.

## Verdict

**RECORD (all 3/3 runs passed).** Official mean training time **21.6s** (22.0 / 21.2 /
21.5), against the previous Track 1 record of 43.9s — a **−51%** cut, far outside
run-to-run spread (the three times span 0.8s), so this is a clear record and not a tie.
All three fresh seeds clear the 57.0% target, the adapter is 13,762,560 params against a
30M cap, and integrity re-checks passed on every run. Code review found the technique
legitimate, the data discipline clean, and the pinned-snapshot handling stricter than the
harness requires.

The record stands with one documented caveat: the accuracy margin is the thinnest yet
(run 3 cleared by ~1.2 problems), so this schedule sits much closer to the target than
its predecessors. That is a legitimate way to spend accuracy headroom for wall-clock
under the current rules, and it is recorded here rather than penalized.

Track 1 progression: 716.9s → 21.6s across six records and three authors — a **33.2×**
cut since the freeze.
