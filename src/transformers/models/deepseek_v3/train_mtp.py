#!/usr/bin/env python
"""tiny_train_mtp.py – REALLY minimal script that trains ONLY `mtp_layer` of a
patched Kimi‑K2 model. We rely on the KL loss that is already computed inside the
model’s forward (between `logits` and `mtp_logits`).

Usage (single‑GPU smoke test):

    accelerate launch train_mtp.py \
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
    AutoModel,
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

    device_map = {'embed_tokens': 0, 'layers.0': 0, 'layers.1': 0, 'layers.2': 0, 'layers.3': 0, 'layers.4': 0, 'layers.5': 0, 'layers.6': 0, 'layers.7': 1, 'layers.8': 1, 'layers.9': 1, 'layers.10': 1, 'layers.11': 1, 'layers.12': 1, 'layers.13': 1, 'layers.14': 2, 'layers.15': 2, 'layers.16': 2, 'layers.17': 2, 'layers.18': 2, 'layers.19': 2, 'layers.20': 2, 'layers.21': 3, 'layers.22': 3, 'layers.23': 3, 'layers.24': 3, 'layers.25': 3, 'layers.26': 3, 'layers.27': 3, 'layers.28': 4, 'layers.29': 4, 'layers.30': 4, 'layers.31': 4, 'layers.32': 4, 'layers.33': 4, 'layers.34': 4, 'layers.35': 5, 'layers.36': 5, 'layers.37': 5, 'layers.38': 5, 'layers.39': 5, 'layers.40': 5, 'layers.41': 5, 'layers.42': 6, 'layers.43': 6, 'layers.44': 6, 'layers.45': 6, 'layers.46': 6, 'layers.47': 6, 'layers.48': 6, 'layers.49': 7, 'layers.50': 7, 'layers.51': 7, 'layers.52': 7, 'layers.53': 7, 'layers.54': 7, 'layers.55': 6, 'layers.56': 5, 'layers.57': 4, 'layers.58': 3, 'layers.59': 2, 'layers.60': 1, 'layers.61': 0, 'norm': 0}
    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, config=hf_config, trust_remote_code=True, device_map=device_map,
                                      torch_dtype=torch.bfloat16)

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
