#!/usr/bin/env python3
"""Experiment B — QLoRA fine-tuning of the 1.5B base (design §7).

Produces experiments/<cell>/adapter/ (PEFT adapter only — the 4-bit base is
never written) plus train_config.json for reproducibility. Overnight-safe:
step checkpoints, --resume, VRAM logged every step group.

v2 hardening (checklist Stage 1): every --eval-steps the trainer evaluates
--val-n validation examples (loss) and runs a --exec-eval-n execution-accuracy
probe with the training model; the probe's best adapter is what gets copied
to adapter/, and examples longer than --max-len abort the run instead of
being silently truncated into partial SQL supervision.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainerCallback)
from trl import SFTConfig, SFTTrainer

from t2sql.datasets import db_stratified_sample
from t2sql.eval.harness import evaluate
from t2sql.inference.engine import extract_sql
from t2sql.prompts.schema import SchemaCache
from t2sql.prompts.template import build_messages, variant_for
from t2sql.training.data import load_examples

ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


class VramLogger(TrainerCallback):
    """Log peak VRAM alongside loss and append fixed-column CSV curves
    (train_log.csv in the run dir — transformers 5.x dropped CSV support)."""

    COLUMNS = ("step", "loss", "lr", "grad_norm", "eval_loss",
               "val_exec_acc", "peak_vram_gb")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if torch.cuda.is_available():
            logs["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        row = {k: logs.get(k, "") for k in self.COLUMNS}
        if not any(row[k] != "" for k in self.COLUMNS if k != "step"):
            return
        row["step"] = state.global_step
        path = Path(args.output_dir) / "train_log.csv"
        new = not path.exists()
        with path.open("a") as fh:
            if new:
                fh.write(",".join(self.COLUMNS) + "\n")
            fh.write(",".join(str(row[k]) for k in self.COLUMNS) + "\n")


class ValProbe(TrainerCallback):
    """Execution-accuracy probe on validation (checklist 1.2-1.3).

    Every eval step: greedy SQL for the first ``n`` val items generated with
    the training model, scored by the gold-side harness (val gold only — the
    sole gold consumer in this script). The best probe's weights are saved to
    ``adapter_best/``; history lands in ``val_metrics.json`` at train end.
    """

    def __init__(self, tokenizer, val_items: list[dict], db_dir: Path,
                 n: int, out_dir: Path, *, prompt_variants: int = 1,
                 few_shot: str = "", schema_samples: bool = False,
                 question_values: bool = False):
        self.tok = tokenizer
        self.items = val_items[:n]
        self.db_dir = db_dir
        self.enabled = n > 0 and bool(self.items)
        self.out_dir = out_dir
        self.best_dir = out_dir / "adapter_best"
        self.prompt_variants = prompt_variants
        self.few_shot = few_shot
        self.schema_samples = schema_samples
        self.schema_cache = SchemaCache(samples=schema_samples,
                                        question_values=question_values)
        self.history: list[dict] = []
        self.best_acc = -1.0
        self.best_step: int | None = None

    def _schema(self, db_id: str, question: str) -> str:
        return self.schema_cache.get(
            self.db_dir / db_id / f"{db_id}.sqlite", question)

    @torch.inference_mode()
    def _probe(self) -> tuple[float, int, int]:
        model, tok = self.model, self.tok
        was_training = model.training
        model.eval()
        # Generation here runs with use_cache disabled (gradient checkpointing
        # requires it during training), so every probe decodes token by token.
        # Restore the cache for the probe and put it back afterwards.
        prev_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True
        preds = []
        try:
            for j, item in enumerate(self.items):
                msgs = build_messages(
                    item["question"], self._schema(item["db_id"],
                                                   item["question"]),
                    few_shot=self.few_shot,
                    variant=variant_for(item["question"], self.prompt_variants))
                ids = chat_ids(tok, msgs, generation_prompt=True)
                x = torch.tensor([ids], device=model.device)
                out = model.generate(x, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                text = tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)
                preds.append({"index": j, "db_id": item["db_id"],
                              "question": item["question"],
                              "prediction": extract_sql(text)})
        finally:
            model.config.use_cache = prev_cache
        if was_training:
            model.train()
        report = evaluate(self.items, preds, self.db_dir)
        return report.accuracy, len(report.scored), report.n_skipped

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.enabled:
            return
        acc, n_scored, n_skipped = self._probe()
        row = {"step": state.global_step, "val_exec_acc": round(acc, 4),
               "n_scored": n_scored, "n_skipped": n_skipped}
        if metrics and "eval_loss" in metrics:
            row["val_loss"] = round(metrics["eval_loss"], 4)
        self.history.append(row)
        if self.trainer is not None:
            self.trainer.log({"val_exec_acc": row["val_exec_acc"]})
        print(f"[probe] step {state.global_step}: val_exec_acc={acc:.3f} "
              f"(n={n_scored}) {row}", flush=True)
        if acc > self.best_acc:
            self.best_acc, self.best_step = acc, state.global_step
            self.model.save_pretrained(self.best_dir)

    def on_train_end(self, args, state, control, **kwargs):
        payload = {"history": self.history, "best_val_exec_acc": self.best_acc
                   if self.best_step is not None else None,
                   "best_step": self.best_step}
        (self.out_dir / "val_metrics.json").write_text(json.dumps(payload, indent=2))



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", default="experiments/B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096,
                    help="abort (no truncation) if any example exceeds this")
    ap.add_argument("--rank", type=int, default=64,
                    help="LoRA rank (the QLoRA paper uses 64 on all linear "
                         "layers; 16 is well below the reference recipe)")
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=0.3,
                    help="the QLoRA paper clips at 0.3; at 1.0 a 14%% share "
                         "of logged steps saturates the clip")
    ap.add_argument("--lr-scheduler", default="cosine",
                    choices=["cosine", "constant"])
    ap.add_argument("--batch-size", type=int, default=1,
                    help="sequences per device step. 1 is fastest per sample "
                         "here (1.71 s vs 2.29 s at bs=2) and the only value "
                         "that fits without gradient checkpointing; effective "
                         "batch is this times --accum")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True,
                    help="trades throughput for activation memory. On by "
                         "default and effectively required: every config with "
                         "it off OOMs on a 4 GB card at this sequence length")
    ap.add_argument("--packing", action="store_true", default=False,
                    help="pack short examples into one sequence. Off by "
                         "default: measured on this GPU it costs 3.2x per "
                         "step (22.0 vs 6.9 s) and buys nothing, because a "
                         "per-device batch of 1 has no padding to reclaim. It "
                         "also triples tokens per optimizer step, silently "
                         "changing the effective batch size.")
    ap.add_argument("--no-packing", dest="packing", action="store_false",
                    help="(default; accepted so recipes stay explicit)")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false",
                    help="escape hatch for larger GPUs; OOMs at 4 GB here")
    ap.add_argument("--no-double-quant", dest="double_quant",
                    action="store_false")
    ap.add_argument("--val-n", type=int, default=200,
                    help="validation examples for the eval-loss diagnostic")
    ap.add_argument("--probe-file", default=None,
                    help="split the execution probe is drawn from; defaults to "
                         "data/splits/holdout.json, which is database-disjoint "
                         "from training, and falls back to val.json")
    ap.add_argument("--exec-eval-n", type=int, default=200,
                    help="probe size, drawn round-robin across databases so it "
                        "does not collapse onto a handful of schemas (0 = off)")
    ap.add_argument("--eval-steps", type=int, default=500,
                    help="probe/eval/save cadence in optimizer steps")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N examples")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: override schedule")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prompt-variants", type=int, default=1,
                    help="instruction variants in play (match build_training_set)")
    ap.add_argument("--few-shot-file", default=None,
                    help="few-shot block used by training data (match build)")
    ap.add_argument("--question-values", action="store_true",
                    help="schema example values chosen to match the question; "
                         "must match build_training_set.py")
    ap.add_argument("--schema-samples", action="store_true",
                    help="row counts + example values in val prompts (match build)")
    return ap.parse_args()


def chat_ids(tokenizer, messages: list[dict], *, generation_prompt: bool) -> list[int]:
    """apply_chat_template -> plain token list (transformers 5.x returns a
    BatchEncoding when tokenize=True)."""
    enc = tokenizer.apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=generation_prompt)
    if not isinstance(enc, list):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):
        enc = enc[0]
    return enc


def check_overflow(examples: list[dict], tokenizer, max_len: int) -> int:
    """Fail fast instead of right-truncating assistant SQL into partial labels."""
    longest, over = 0, 0
    for ex in examples:
        n = len(chat_ids(tokenizer, ex["messages"], generation_prompt=False))
        longest = max(longest, n)
        if n > max_len:
            over += 1
    if over:
        raise SystemExit(
            f"{over}/{len(examples)} examples exceed --max-len={max_len} "
            f"(longest {longest}); raise --max-len — truncation would cut the "
            f"gold SQL tail and corrupt the supervision.")
    return longest


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_path = ROOT / "data" / "splits" / "train_sft.json"
    examples = load_examples(examples_path)
    if args.limit:
        examples = examples[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Abort rather than silently right-truncate: a clipped assistant turn
    # teaches a partial query. Costs one tokenization pass over the set.
    longest = check_overflow(examples, tokenizer, args.max_len)

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=args.double_quant,
        ),
        device_map="auto",
    )
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank,
        lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    ))
    model.print_trainable_parameters()

    # validation set: eval loss on gold assistant turns + execution probe on
    # gold-free prompts (built by the probe itself from the raw items)
    val_raw = json.loads((ROOT / "data" / "splits" / "val.json").read_text()) \
        [: args.val_n] if args.val_n > 0 else []
    val_messages: list[dict] = []
    few_shot = Path(args.few_shot_file).read_text() if args.few_shot_file else ""
    if val_raw:
        from t2sql.training.data import build_training_examples
        val_messages = build_training_examples(
            val_raw, ROOT / "data" / "spider" / "database",
            prompt_variants=args.prompt_variants, few_shot=few_shot,
            samples=args.schema_samples,
            question_values=args.question_values)

    # The probe measures generalization, so it runs on databases the model
    # never trained on. val.json shares all 140 databases with training, so
    # it cannot answer that question; holdout.json is carved by whole database.
    probe_path = (ROOT / "data" / "splits" / args.probe_file if args.probe_file
                  else ROOT / "data" / "splits" / "holdout.json")
    if not probe_path.exists():
        probe_path = ROOT / "data" / "splits" / "val.json"
    probe_items: list[dict] = []
    if probe_path.exists() and args.exec_eval_n > 0:
        pool = json.loads(probe_path.read_text())
        probe_items = db_stratified_sample(pool, args.exec_eval_n, seed=args.seed)
        n_dbs = len({x["db_id"] for x in probe_items})
        print(f"probe: {len(probe_items)} items over {n_dbs} databases from "
              f"{probe_path.name}", flush=True)
        train_dbs = {x["db_id"] for x in json.loads(
            (ROOT / "data" / "splits" / "train.json").read_text())}
        shared = train_dbs & {x["db_id"] for x in probe_items}
        if shared:
            print(f"WARNING: {len(shared)} probe databases also appear in "
                  f"training; this probe measures recall, not generalization",
                  flush=True)

    conf = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=100,
        optim="paged_adamw_8bit",
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_len,
        assistant_only_loss=True,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=args.packing,
        logging_steps=5,
        eval_strategy="steps" if val_messages else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if val_messages else "epoch",
        save_steps=args.eval_steps,
        save_total_limit=3,
        seed=args.seed,
        bf16=True,
        report_to=[],   # curves go to train_log.csv via VramLogger
    )
    probe = ValProbe(tokenizer, probe_items,
                     ROOT / "data" / "spider" / "database",
                     len(probe_items), out_dir,
                     prompt_variants=args.prompt_variants, few_shot=few_shot,
                     schema_samples=args.schema_samples,
                     question_values=args.question_values)
    trainer = SFTTrainer(
        model=model,
        args=conf,
        train_dataset=Dataset.from_list(examples),
        eval_dataset=Dataset.from_list(val_messages) if val_messages else None,
        processing_class=tokenizer,
        callbacks=[VramLogger(), probe],
    )
    probe.model = trainer.model
    probe.trainer = trainer

    resume_ckpt = None
    if args.resume:
        ckpts = sorted(out_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        resume_ckpt = str(ckpts[-1]) if ckpts else None

    started = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - started

    adapter_dir = out_dir / "adapter"
    if (out_dir / "adapter_best" / "adapter_config.json").exists():
        shutil.copytree(out_dir / "adapter_best", adapter_dir, dirs_exist_ok=True)
        adapter_source = "best_val_exec_acc"
    else:
        model.save_pretrained(adapter_dir)
        adapter_source = "final_step"
    tokenizer.save_pretrained(adapter_dir)
    # Recorded from the live arguments, so the artifact cannot drift from the
    # run that produced it.
    (out_dir / "train_config.json").write_text(json.dumps({
        **vars(args),
        "n_examples": len(examples),
        "n_val_examples": len(val_messages),
        "probe_items": len(probe_items),
        "probe_databases": len({x["db_id"] for x in probe_items}),
        "probe_source": probe_path.name,
        "effective_batch": args.batch_size * args.accum,
        "longest_example_tokens": longest,
        "global_steps": trainer.state.global_step,
        "train_runtime_s": round(elapsed),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "adapter_selection": adapter_source,
        "lora": {"r": args.rank, "alpha": 2 * args.rank,
                 "dropout": args.lora_dropout,
                 "target_modules": TARGET_MODULES},
    }, indent=2))
    print(f"saved adapter -> {adapter_dir} ({adapter_source}, "
          f"{trainer.state.global_step} steps, {elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
