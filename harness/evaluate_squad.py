"""Fixed SQuAD v1.1 evaluator for Track 2 — the measuring stick. Do not vary.

Protocol (frozen in spec-t2.yaml): 0-shot prompt "Context: {c}\nQuestion: {q}\nAnswer:",
greedy, bf16, max_new_tokens=32, batch 128. Score = exact match against ANY gold answer
using the official SQuAD normalization (lowercase, strip punctuation/articles/whitespace).

Usage:
  LORA_SPEEDRUN_SPEC=spec-t2.yaml python harness/evaluate_squad.py \
      --data-dir data/squad_test [--adapter-dir RUN_DIR] [--limit 128] [--out eval.json]
"""

import argparse
import json
import os
import re
import string
import time
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = Path(os.environ.get("LORA_SPEEDRUN_SPEC", REPO_ROOT / "spec-t2.yaml"))
SPEC = yaml.safe_load((REPO_ROOT / SPEC_PATH if not SPEC_PATH.is_absolute() else SPEC_PATH).read_text())


def normalize(text: str) -> str:
    """Official SQuAD answer normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_pred(generation: str) -> str:
    """First line of the generation, truncated at any hallucinated continuation."""
    text = generation.split("Context:")[0].split("Question:")[0]
    return text.strip().split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="load_from_disk dir of the eval split")
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--base-model", default=SPEC["base_model"])
    ap.add_argument("--batch-size", type=int, default=SPEC["eval"]["batch_size"])
    ap.add_argument("--max-new-tokens", type=int, default=SPEC["eval"]["max_new_tokens"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda")
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    ds = load_from_disk(args.data_dir)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    prompts = [SPEC["eval"]["prompt_format"].format(context=c, question=q)
               for c, q in zip(ds["context"], ds["question"])]
    golds = [[normalize(t) for t in a["text"]] for a in ds["answers"]]

    n_correct, t0 = 0, time.monotonic()
    with torch.inference_mode():
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(model.device)
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id,
                stop_strings=["Context:", "Question:"], tokenizer=tok)
            gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, gen in enumerate(gens):
                if normalize(extract_pred(gen)) in golds[i + j]:
                    n_correct += 1
            done = i + len(batch)
            print(f"  eval {done}/{len(prompts)}  running acc={n_correct/done:.4f}", flush=True)

    result = {
        "accuracy": n_correct / len(prompts),
        "n_correct": n_correct,
        "n_total": len(prompts),
        "adapter_dir": args.adapter_dir,
        "base_model": args.base_model,
        "eval_seconds": round(time.monotonic() - t0, 1),
        "protocol": SPEC["eval"],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "protocol"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
