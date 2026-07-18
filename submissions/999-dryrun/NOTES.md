# Submission notes — 999-dryrun

**What this is:** an end-to-end test of the submission pipeline (CI static validation,
the automated Claude security screen, and the `/verify` gating logic) before the
benchmark opens to outside submissions. The training script is byte-identical to
`submissions/000-baseline/train.py`.

This PR also deliberately touches a file outside `submissions/` so that commenting
`/verify` exercises the refusal path — the workflow must decline to auto-run and ask
for manual review instead.

**This is not a record attempt and will be closed without merging.**
