#!/usr/bin/env python3
"""Experiment B — QLoRA fine-tuning of the 1.5B base (design §7).

Produces experiments/<cell>/adapter/ (PEFT adapter only — the 4-bit base is
never written) plus train_config.json for reproducibility. Overnight-safe:
epoch checkpoints, --resume, VRAM logged every step group.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from t2sql.training.data import build_training_examples, load_examples

ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


class VramLogger(TrainerCallback):
    """Log peak VRAM alongside loss."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            logs["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", default="experiments/B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="covers 98.8% of examples (p95=1372, max=2352 tokens)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N examples")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: override schedule")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_path = ROOT / "data" / "splits" / "train_sft.json"
    examples = load_examples(examples_path)
    if args.limit:
        examples = examples[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    ))
    model.print_trainable_parameters()

    conf = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,   # ~3% of the 2-epoch schedule (6300/4 x 2 = 3150 steps)
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        max_length=args.max_len,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        bf16=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=conf,
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
        callbacks=[VramLogger()],
    )

    resume_ckpt = None
    if args.resume:
        ckpts = sorted(out_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        resume_ckpt = str(ckpts[-1]) if ckpts else None

    started = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - started

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (out_dir / "train_config.json").write_text(json.dumps({
        **vars(args),
        "n_examples": len(examples),
        "global_steps": trainer.state.global_step,
        "train_runtime_s": round(elapsed),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": TARGET_MODULES},
    }, indent=2))
    print(f"saved adapter -> {adapter_dir} "
          f"({trainer.state.global_step} steps, {elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()