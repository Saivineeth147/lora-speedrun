<!-- For record attempts, title format: [record] <handle> — <one-line technique> — <mm:ss self-reported> -->

## Type

- [ ] Record attempt
- [ ] Notable attempt (novel technique, doesn't beat the record)
- [ ] Harness / docs / other

## Record attempts — self-verification

Run `python harness/run_submission.py submissions/NNN-you --runs 3` on a 4090 and paste:

| run | seed | train wall-clock | GSM8K acc |
|-----|------|------------------|-----------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

- GPU used: <!-- 4090 strongly preferred; other 24GB cards accepted for triage only -->
- Extra pip packages (must match config.yaml): 

## Checklist

- [ ] I read [TASK.md](../blob/main/TASK.md) — training data is the GSM8K train split, nothing else
- [ ] `train.py` follows the contract (`--data-dir`, `--output-dir`, `--seed`) and runs offline
- [ ] No model weights committed
- [ ] `NOTES.md` explains the mechanism, not just the result
- [ ] I understand a maintainer reruns this 3× with fresh seeds before any merge
