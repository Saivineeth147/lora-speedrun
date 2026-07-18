"""Fixed GSM8K evaluator — the measuring stick for every submission. Do not vary.

Protocol (frozen in spec.yaml): 0-shot prompt "Question: {q}\nAnswer:", greedy decoding,
bf16, max_new_tokens=512, batch 64, generation truncated at any hallucinated "Question:",
answer = last `#### <number>` match, numeric exact-match against gold.

Usage:
  python harness/evaluate_gsm8k.py --data-dir data/gsm8k_test [--adapter-dir RUN_DIR]
                                   [--limit 32] [--out eval.json]
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = yaml.safe_load((REPO_ROOT / "spec.yaml").read_text())
ANSWER_RE = re.compile(SPEC["eval"]["answer_regex"])


def extract_pred(generation: str):
    """Last '#### <number>' in the generation, truncated at hallucinated next questions."""
    text = generation.split("Question:")[0]
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_gold(answer_field: str) -> float:
    return float(answer_field.split("####")[-1].strip().replace(",", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="load_from_disk dir of the test split")
    ap.add_argument("--adapter-dir", default=None, help="PEFT adapter dir (omit = base model)")
    ap.add_argument("--base-model", default=SPEC["base_model"])
    ap.add_argument("--batch-size", type=int, default=SPEC["eval"]["batch_size"])
    ap.add_argument("--max-new-tokens", type=int, default=SPEC["eval"]["max_new_tokens"])
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N (smoke test)")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--dump-preds", default=None, help="write per-item predictions JSONL here")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    ds = load_from_disk(args.data_dir)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    prompts = [SPEC["eval"]["prompt_format"].format(question=q) for q in ds["question"]]
    golds = [extract_gold(a) for a in ds["answer"]]

    n_correct, records, t0 = 0, [], time.monotonic()
    with torch.inference_mode():
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
                stop_strings=["Question:"],
                tokenizer=tok,
            )
            gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, gen in enumerate(gens):
                pred = extract_pred(gen)
                gold = golds[i + j]
                ok = pred is not None and abs(pred - gold) < 1e-4
                n_correct += ok
                records.append({"idx": i + j, "pred": pred, "gold": gold, "correct": ok})
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
    if args.dump_preds:
        with open(args.dump_preds, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
