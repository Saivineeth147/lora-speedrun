"""Liger Kernel with a larger microbatch on top of record #1.

Keeps the record's data pipeline and training hyperparameters unchanged while
using Liger's fused Triton kernels and increasing work per GPU launch without
changing the effective batch size.

Contract: python train.py --data-dir <gsm8k_train> --output-dir <dir> --seed <int>
"""

import argparse

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
PACK_LEN = 1024


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
    eos = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    ds = load_from_disk(args.data_dir)

    # Completion-only masking: mask the "Question: ...\nAnswer:" prompt, train
    # only on " {answer}<eos>". add_special_tokens=False so no BOS is injected
    # mid-pack.
    def encode(ex):
        p_ids = tok(f"Question: {ex['question']}\nAnswer:",
                    add_special_tokens=False)["input_ids"]
        c_ids = tok(f" {ex['answer']}",
                    add_special_tokens=False)["input_ids"] + [eos]
        return {"ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids}

    enc = ds.map(encode, remove_columns=ds.column_names, desc="encode")

    # Greedy packing into dense PACK_LEN blocks.
    blocks_i, blocks_l, buf_i, buf_l = [], [], [], []
    for row in enc:
        ids, labels = row["ids"][:PACK_LEN], row["labels"][:PACK_LEN]
        if buf_i and len(buf_i) + len(ids) > PACK_LEN:
            blocks_i.append(buf_i); blocks_l.append(buf_l); buf_i, buf_l = [], []
        buf_i += ids; buf_l += labels
    if buf_i:
        blocks_i.append(buf_i); blocks_l.append(buf_l)
    packed = Dataset.from_dict({"input_ids": blocks_i, "labels": blocks_l})
    print(f"packed {len(enc)} examples -> {len(packed)} blocks of <= {PACK_LEN} tokens")

    def collate(features):
        m = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            pad = m - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attn),
                "labels": torch.tensor(labels)}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
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
            use_liger_kernel=True,
        ),
        train_dataset=packed,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
