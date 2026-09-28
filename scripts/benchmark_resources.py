#!/usr/bin/env python3
"""Resource benchmarks (design §16): load time, latency, throughput, VRAM/RAM,
disk footprint — quantized vs unquantized vs CPU. Run with the GPU IDLE (after
Phase 10) so numbers are uncontended. Writes results/resource_usage/benchmarks.json.

Modeled configs: 4-bit base (cell A/B inference), 4-bit + adapter (cells B-D),
bf16 base (unquantized comparison). CPU phase runs in an isolated subprocess
(CUDA hidden) so device_map lands on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def dir_size_gb(path: Path) -> float:
    if not path or not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 2 ** 30, 3)


def latency_sample(llm, items, n: int, max_new_tokens: int) -> dict:
    from t2sql.prompts.schema import schema_text
    from t2sql.prompts.template import build_messages, full_question

    rng = random.Random(0)
    sample = [items[i] for i in rng.sample(range(len(items)), min(n, len(items)))]
    schema_cache: dict[str, str] = {}
    times, out_tokens = [], []
    for item in sample:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = schema_text(
                ROOT / "data" / "spider" / "database" / db_id / f"{db_id}.sqlite")
        messages = build_messages(full_question(item), schema_cache[db_id])
        t0 = time.perf_counter()
        text = llm.generate(messages, max_new_tokens=max_new_tokens)
        times.append(time.perf_counter() - t0)
        out_tokens.append(len(llm.tokenizer(
            text, add_special_tokens=False)["input_ids"]))
    times_sorted = sorted(times)
    p = lambda q: round(times_sorted[min(len(times_sorted) - 1,
                                         round(q * (len(times_sorted) - 1)))], 3)
    total_s = sum(times)
    return {
        "n": len(sample), "max_new_tokens": max_new_tokens,
        "mean_s": round(total_s / len(times), 3),
        "p50_s": p(0.5), "p95_s": p(0.95),
        "output_tokens_per_s": round(sum(out_tokens) / total_s, 2),
    }


def run_config(name: str, quantize: bool, adapter: str | None, items, n: int) -> dict:
    import torch
    from t2sql.inference.engine import LLM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM("Qwen/Qwen2.5-Coder-1.5B-Instruct", quantize=quantize, adapter=adapter)
    load_s = round(time.perf_counter() - t0, 1)
    lat = latency_sample(llm, items, n, max_new_tokens=128)
    out = {
        "config": name, "load_s": load_s,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        "process_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF)
                                     .ru_maxrss / 2 ** 20, 3),
        **lat,
    }
    del llm
    torch.cuda.empty_cache()
    return out


def cpu_phase(n: int) -> dict:
    from t2sql.inference.engine import LLM
    items = json.loads((ROOT / "data" / "splits" / "val.json").read_text())
    t0 = time.perf_counter()
    llm = LLM("Qwen/Qwen2.5-Coder-1.5B-Instruct", quantize=False)
    load_s = round(time.perf_counter() - t0, 1)
    lat = latency_sample(llm, items, n, max_new_tokens=64)
    return {"config": "fp32_cpu", "load_s": load_s,
            "process_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF)
                                         .ru_maxrss / 2 ** 20, 3), **lat}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="full", choices=["full", "cpu"])
    ap.add_argument("--n", type=int, default=20, help="latency sample size")
    ap.add_argument("--n-cpu", type=int, default=5)
    ap.add_argument("--out", default="results/resource_usage/benchmarks.json")
    args = ap.parse_args()

    if args.mode == "cpu":
        print(json.dumps(cpu_phase(args.n_cpu)))
        return

    from t2sql.inference.engine import LLM
    import torch

    items = json.loads((ROOT / "data" / "splits" / "val.json").read_text())
    adapter = str(ROOT / "experiments" / "B" / "adapter")
    free, total = torch.cuda.mem_get_info()
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30

    model_dir = LLM.cached_model_dir("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {"ram_total_gb": round(ram_gb, 2),
                   "vram_total_gb": round(total / 2 ** 30, 2),
                   "torch": torch.__version__},
        "disk_gb": {
            "hf_model_cache": dir_size_gb(model_dir.parent.parent if model_dir else None),
            "lora_adapter": dir_size_gb(ROOT / "experiments" / "B"),
            "data_spider": dir_size_gb(ROOT / "data" / "spider"),
            "data_bird": dir_size_gb(ROOT / "data" / "bird"),
            "data_sparc": dir_size_gb(ROOT / "data" / "sparc"),
        },
        "gpu_configs": [
            run_config("4bit_base", True, None, items, args.n),
            run_config("4bit_base_adapter", True, adapter, items, args.n),
            run_config("bf16_base", False, None, items, min(args.n, 10)),
        ],
    }

    proc = subprocess.run([sys.executable, __file__, "--mode", "cpu",
                           "--n-cpu", str(args.n_cpu)],
                          capture_output=True, text=True, env={**os.environ,
                                                               "CUDA_VISIBLE_DEVICES": ""},
                          cwd=ROOT, timeout=3600)
    if proc.returncode == 0 and proc.stdout.strip():
        results["cpu_config"] = json.loads(proc.stdout.strip().splitlines()[-1])
    else:
        results["cpu_config"] = {"error": proc.stderr[-500:]}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()