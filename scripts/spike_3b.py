#!/usr/bin/env python3
"""3B QLoRA feasibility spike (design §5): load 3B 4-bit + LoRA + one real
optimizer step at the production training shape (seq 1024, batch 1).

Go/no-go for experiment cell B2: peak VRAM must stay under HEADROOM_GiB and
no OOM may occur. Uses a synthetic token batch — this measures memory, not
quality, so no data or gold SQL is involved.
"""
from __future__ import annotations

import gc
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
SEQ_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
HEADROOM_GIB = 3.85


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    free, total = torch.cuda.mem_get_info()
    print(f"before: {free / 2**30:.2f} GiB free of {total / 2**30:.2f}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="auto",
    )
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # production training shape uses gradient checkpointing (design §7)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    vocab = model.config.vocab_size
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (1, SEQ_LEN), device=model.device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )

    try:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(f"VERDICT: REJECTED — OOM during step: {str(e)[:120]}")
        raise SystemExit(1)

    peak = torch.cuda.max_memory_allocated() / 2**30
    verdict = "COMMITTED" if peak < HEADROOM_GIB else "REJECTED (too close to limit)"
    print(f"peak VRAM: {peak:.2f} GiB (limit {HEADROOM_GIB})")
    print(f"VERDICT: {verdict}")
    if verdict.startswith("REJECTED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()