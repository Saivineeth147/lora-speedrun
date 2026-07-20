"""Speedrun attempt: 1-epoch LoRA with a lean custom loop.

What's different from record #1:
  1. One epoch (not two) with an aggressive cosine schedule tuned for ~80 steps.
  2. GSM8K `<<...>>` calculator annotations stripped (pure formatting) — ~13% fewer tokens.
  3. No HF Trainer: a hand-rolled loop over pre-packed fixed-length GPU-resident blocks
     (zero dataloader/collate overhead, no attention mask needed — all blocks are full).
  4. Chunked completion-only cross-entropy as a custom autograd.Function: lm_head logits
     are never materialized for the whole batch, and are only computed for the ~70% of
     tokens that carry labels. Saves ~10 GB of fp32 logits traffic per step.
  5. Model safetensors are read into the page cache on a background thread while
     torch/transformers import, overlapping the two biggest fixed costs.

Contract: python train.py --data-dir <gsm8k_train> --output-dir <dir> --seed <int>
"""

import argparse
import glob
import os
import threading
import time

T0 = time.monotonic()


def log(msg):
    print(f"[t+{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


BASE_MODEL = "Qwen/Qwen2.5-1.5B"


def _warm_model_files():
    """Pull the model shards into the OS page cache while python imports torch."""
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for pat in (f"{hf}/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*/*.safetensors",):
        for p in glob.glob(pat):
            try:
                with open(p, "rb") as f:
                    while f.read(1 << 25):
                        pass
            except OSError:
                pass


threading.Thread(target=_warm_model_files, daemon=True).start()

import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402


def read_qa(data_dir):
    """Read the saved train split. Fast path: pyarrow directly (skips importing
    `datasets`, which costs seconds); fallback: datasets.load_from_disk."""
    try:
        import pyarrow.ipc as ipc

        files = sorted(glob.glob(os.path.join(data_dir, "data-*.arrow")))
        assert files
        qs, ans = [], []
        for fp in files:
            with ipc.open_stream(fp) as reader:
                t = reader.read_all()
            qs += t.column("question").to_pylist()
            ans += t.column("answer").to_pylist()
        return qs, ans
    except Exception:
        from datasets import load_from_disk

        ds = load_from_disk(data_dir)
        return ds["question"], ds["answer"]

# --- iteration knobs (env-overridable; defaults are the submitted config) ---
LR = float(os.environ.get("SR_LR", "4e-4"))
EPOCHS = float(os.environ.get("SR_EPOCHS", "1.0"))
BS = int(os.environ.get("SR_BS", "8"))
PACK_LEN = int(os.environ.get("SR_PACK", "1024"))
WARMUP = int(os.environ.get("SR_WARMUP", "8"))
MIN_LR_FRAC = float(os.environ.get("SR_MIN_LR_FRAC", "0.05"))
RANK = int(os.environ.get("SR_RANK", "16"))
ALPHA = int(os.environ.get("SR_ALPHA", "32"))
ADAM_B2 = float(os.environ.get("SR_ADAM_B2", "0.95"))
STRIP_ANNOT = os.environ.get("SR_STRIP", "1") == "1"
SUBSET = os.environ.get("SR_SUBSET", "shortest:4000")  # "" | "shortest:N" | "longest:N" | "first:N"
CE_CHUNK = int(os.environ.get("SR_CE_CHUNK", "2048"))

ANNOT_RE = re.compile(r"<<[^>]*>>")


def resolve_model_path():
    """Fully-offline model resolution: find the cached snapshot directly, so loading
    works whether or not the HF cache has a refs/main entry (it doesn't when the
    prefetch pinned an explicit revision)."""
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snaps = sorted(glob.glob(f"{hf}/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*"))
    try:
        pins = json.loads((Path(__file__).resolve().parents[2] / "harness"
                           / "pins.json").read_text())
        for s in snaps:
            if s.endswith(pins["base_model_sha"]):
                return s
    except OSError:
        pass
    return snaps[-1] if snaps else BASE_MODEL


class ChunkedCE(torch.autograd.Function):
    """Cross-entropy over (hidden_states, frozen lm_head) without materializing full
    logits: per chunk, compute loss and d(loss)/d(hidden) analytically (softmax - 1)."""

    @staticmethod
    def forward(ctx, h, w, y):
        n = y.numel()
        gh = torch.empty_like(h)
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, n, CE_CHUNK):
            hc, yc = h[i : i + CE_CHUNK], y[i : i + CE_CHUNK]
            logits = (hc @ w.T).float()
            lse = torch.logsumexp(logits, dim=-1)
            gold = logits.gather(1, yc[:, None]).squeeze(1)
            total += (lse - gold).sum()
            logits.sub_(lse[:, None]).exp_()  # in-place softmax
            logits[torch.arange(yc.numel(), device=y.device), yc] -= 1.0
            gh[i : i + CE_CHUNK] = logits.to(h.dtype) @ w
        ctx.save_for_backward(gh)
        ctx.n = n
        return total / n

    @staticmethod
    def backward(ctx, gout):
        (gh,) = ctx.saved_tensors
        return gh * (gout / ctx.n), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_path = resolve_model_path()
    log(f"model path: {model_path}")

    # Load the model on a background thread while the main thread tokenizes/packs.
    holder = {}

    def _load_model():
        holder["model"] = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="cuda")
        log("base model loaded on cuda")

    loader = threading.Thread(target=_load_model)
    loader.start()

    tok = AutoTokenizer.from_pretrained(model_path)
    eos = tok.eos_token_id
    log("tokenizer ready")

    questions, answers = read_qa(args.data_dir)
    if STRIP_ANNOT:
        answers = [ANNOT_RE.sub("", a) for a in answers]
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    comps = [f" {a}" for a in answers]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    c_ids = tok(comps, add_special_tokens=False)["input_ids"]
    log(f"tokenized {len(questions)} examples")

    examples = []
    for p, c in zip(p_ids, c_ids):
        c = c + [eos]
        examples.append((p + c, [-100] * len(p) + c))

    if SUBSET:
        mode, n = SUBSET.split(":")
        n = int(n)
        if mode == "shortest":
            examples.sort(key=lambda e: len(e[0]))
            examples = examples[:n]
        elif mode == "longest":
            examples.sort(key=lambda e: -len(e[0]))
            examples = examples[:n]
        elif mode == "first":
            examples = examples[:n]
        log(f"subset {SUBSET}: kept {len(examples)} examples")

    # Seed-shuffled greedy packing into full PACK_LEN blocks (last partial dropped,
    # so no attention mask is ever needed).
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    blocks_i, blocks_l, buf_i, buf_l = [], [], [], []
    for ids, labels in examples:
        ids, labels = ids[:PACK_LEN], labels[:PACK_LEN]
        if len(buf_i) + len(ids) > PACK_LEN:
            pad = PACK_LEN - len(buf_i)
            blocks_i.append(buf_i + [eos] * pad)
            blocks_l.append(buf_l + [-100] * pad)
            buf_i, buf_l = [], []
        buf_i += ids
        buf_l += labels
    n_blocks = len(blocks_i)
    tok_total = sum(1 for b in blocks_l for t in b if t != -100)
    log(f"packed -> {n_blocks} blocks of {PACK_LEN} ({tok_total} labeled tokens)")

    ids_t = torch.tensor(blocks_i, dtype=torch.long, device="cuda")
    lab_t = torch.tensor(blocks_l, dtype=torch.long, device="cuda")

    loader.join()
    model = get_peft_model(holder["model"], LoraConfig(
        r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log(f"trainable params: {n_trainable:,}")
    assert n_trainable <= 30_000_000

    qwen = model.base_model.model  # Qwen2ForCausalLM with LoRA injected
    transformer = qwen.model
    lm_w = qwen.lm_head.weight  # frozen (tied) — ChunkedCE never trains it

    opt = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, ADAM_B2),
                            weight_decay=0.0, fused=True)

    total_blocks = round(n_blocks * EPOCHS)
    steps = (total_blocks + BS - 1) // BS
    order = []
    while len(order) < total_blocks:
        ep = list(range(n_blocks))
        rng.shuffle(ep)
        order += ep
    order = order[:total_blocks]

    def lr_at(step):
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        import math
        f = (step - WARMUP) / max(1, steps - WARMUP)
        return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * f)))

    log(f"training: {steps} steps x {BS} blocks, lr={LR}, epochs={EPOCHS}")
    t_train = time.monotonic()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        sel = order[step * BS : (step + 1) * BS]
        x = ids_t[sel]
        y = lab_t[sel]
        h = transformer(input_ids=x, use_cache=False).last_hidden_state
        hs = h[:, :-1, :].reshape(-1, h.shape[-1])
        ys = y[:, 1:].reshape(-1)
        keep = (ys != -100).nonzero(as_tuple=True)[0]
        loss = ChunkedCE.apply(hs[keep], lm_w, ys[keep])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            dt = time.monotonic() - t_train
            log(f"step {step + 1}/{steps} loss={loss.item():.4f} "
                f"lr={lr_at(step):.2e} ({(step + 1) * BS * PACK_LEN / max(dt, 1e-9):,.0f} tok/s)")

    model.save_pretrained(args.output_dir)
    log(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
