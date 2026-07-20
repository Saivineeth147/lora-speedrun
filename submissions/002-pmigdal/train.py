"""Track 1 attempt: one aggressive epoch over short dense packs.

Keeps record #1's completion-only objective, but uses one higher-LR pass over
all training examples and packs into 512-token sequences rather than 1024.
The per-device batch is doubled so each microbatch still carries about 4096
tokens. Shorter sequences reduce causal attention work and memory pressure.
"""

import argparse

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
BASE_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
PACK_LEN = 512
TRAIN_EXAMPLES = 3000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Resolve the frozen snapshot directly. The verifier's offline cache contains
    # this pinned commit but does not necessarily contain a mutable `main` ref.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    ds = load_from_disk(args.data_dir).select(range(TRAIN_EXAMPLES))

    def encode(ex):
        prompt = tok(f"Question: {ex['question']}\nAnswer:",
                     add_special_tokens=False)["input_ids"]
        completion = tok(f" {ex['answer']}",
                         add_special_tokens=False)["input_ids"] + [eos]
        return {
            "ids": prompt + completion,
            "labels": [-100] * len(prompt) + completion,
        }

    encoded = ds.map(encode, remove_columns=ds.column_names, desc="encode")

    block_ids, block_labels, ids_buf, labels_buf = [], [], [], []
    truncated = 0
    for row in encoded:
        if len(row["ids"]) > PACK_LEN:
            truncated += 1
        ids = row["ids"][:PACK_LEN]
        labels = row["labels"][:PACK_LEN]
        if ids_buf and len(ids_buf) + len(ids) > PACK_LEN:
            block_ids.append(ids_buf)
            block_labels.append(labels_buf)
            ids_buf, labels_buf = [], []
        ids_buf += ids
        labels_buf += labels
    if ids_buf:
        block_ids.append(ids_buf)
        block_labels.append(labels_buf)
    packed = Dataset.from_dict({"input_ids": block_ids, "labels": block_labels})
    print(f"packed {len(encoded)} examples into {len(packed)} <= {PACK_LEN}-token "
          f"blocks; truncated {truncated} overlength examples")

    def collate(features):
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            ids = feature["input_ids"]
            labs = feature["labels"]
            padding = width - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(labs + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=4e-4,
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
        train_dataset=packed,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
