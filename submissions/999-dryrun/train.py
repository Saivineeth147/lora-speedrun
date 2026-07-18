"""Baseline: plain LoRA SFT on GSM8K train. Deliberately naive — beat it.

Implements the submission contract:
  python train.py --data-dir <gsm8k_train> --output-dir <dir> --seed <int>
"""

import argparse

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
MAX_LEN = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    ds = load_from_disk(args.data_dir)

    def tokenize(ex):
        enc = tok(f"Question: {ex['question']}\nAnswer: {ex['answer']}",
                  truncation=True, max_length=MAX_LEN - 1)
        enc["input_ids"].append(tok.eos_token_id)
        enc["attention_mask"].append(1)
        return enc

    ds = ds.map(tokenize, remove_columns=ds.column_names, desc="tokenize")

    # pad==eos for Qwen, so mask labels via attention_mask (not pad id) — otherwise
    # the model never learns to stop and eval generations run long.
    def collate(features):
        batch = tok.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            dataloader_num_workers=2,
        ),
        train_dataset=ds,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
