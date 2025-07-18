#!/usr/bin/env python
"""tiny_train_mtp.py – REALLY minimal script that trains ONLY `mtp_layer` of a
patched Kimi‑K2 model. We rely on the KL loss that is already computed inside the
model’s forward (between `logits` and `mtp_logits`).

Usage (single‑GPU smoke test):

    accelerate launch tiny_train_mtp.py \
        --model_path /data/weights/vllm-moonshotai--Kimi-K2-Instruct/003/ \
        --output_dir ./mtp_out
"""

import os
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

class TinyDataset(Dataset):
    """Wrap raw text -> chat prompt -> token ids."""

    def __init__(self, texts: List[str], tok, max_len: int = 1024):
        self.texts = texts
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tok(t, truncation=True, max_length=self.max_len, padding=False)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"])),
            "labels": enc["input_ids"],  # loop labels back so model’s KL loss triggers
        }

class PadCollator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id

    def __call__(self, feats: List[Dict]):
        max_len = max(len(f["input_ids"]) for f in feats)
        batch = {k: [] for k in feats[0].keys()}
        for f in feats:
            pad_n = max_len - len(f["input_ids"])
            for k, v in f.items():
                batch[k].append(v + [self.pad] * pad_n)
        return {k: torch.tensor(v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./mtp_out")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--slice", type=str, default="train[:1%]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = DeepseekV3ForCausalLM.from_pretrained(args.model_path, config=hf_config, trust_remote_code=True)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.model.mtp_layer.parameters():
        p.requires_grad = True

    # tiny dataset
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.slice)
    texts = [t for t in raw["text"] if t.strip()][:100]  # hard‑limit 100 examples
    ds = TinyDataset(texts, tokenizer, args.max_len)
    collator = PadCollator(tokenizer)

    # training
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        logging_steps=10,
        save_total_limit=1,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "mtp_adapter"))

if __name__ == "__main__":
    main()
