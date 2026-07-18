"""Your record attempt. Start from submissions/000-baseline/train.py (a complete
working implementation) and make it faster.

The contract the harness enforces (see TASK.md):

  python train.py --data-dir <gsm8k_train_dir> --output-dir <dir> --seed <int>

- --data-dir : datasets.load_from_disk()-loadable dir containing ONLY the train split
- --seed     : chosen by the verifier; consume it — all 3 fresh-seed runs must pass
- on exit    : <output-dir> contains a standard PEFT adapter
               (adapter_config.json + adapter_model.safetensors, <= 30M params total)
- runs offline (HF_HUB_OFFLINE=1): everything you need must be pre-cached or shipped here
- the whole process is timed: startup, tokenization, training, saving — all of it
"""

raise SystemExit(
    "TEMPLATE: copy submissions/000-baseline/train.py here as your starting point."
)
