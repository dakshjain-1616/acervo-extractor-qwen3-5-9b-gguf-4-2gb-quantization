#!/usr/bin/env python3
"""
03_custom_config.py — Customise behaviour via env vars and custom prompt files.

Demonstrates:
  • Overriding defaults with environment variables before import
  • Loading custom prompts from a file with load_prompts()
  • Writing a temporary prompts file and using it in a dry-run benchmark
  • Reading effective configuration back from the module
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from pathlib import Path

# Set env vars BEFORE importing the module so they take effect on module-level constants
os.environ.setdefault("NUM_PROMPTS", "5")
os.environ.setdefault("MAX_NEW_TOKENS", "20")
os.environ.setdefault("WARMUP_RUNS", "1")

from acervo_extractor_qwe.benchmark import (
    load_prompts,
    estimate_quantized_metrics,
    generate_markdown_report,
    NUM_PROMPTS,
    MAX_NEW_TOKENS,
    WARMUP_RUNS,
)
from acervo_extractor_qwe.memory_estimator import format_markdown_table, get_available_ram_gb

# --- Show effective config ---
print("=== Effective Configuration ===")
print(f"  NUM_PROMPTS    : {NUM_PROMPTS}")
print(f"  MAX_NEW_TOKENS : {MAX_NEW_TOKENS}")
print(f"  WARMUP_RUNS    : {WARMUP_RUNS}")

# --- Custom prompts via temp file ---
custom_prompts_text = "\n".join([
    "Extract the invoice number and total amount from this document:",
    "List all party names mentioned in the following contract:",
    "Identify the governing law clause in this legal text:",
    "What currency and amounts are referenced in this financial statement?",
    "Summarise the termination conditions in this agreement:",
])

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write(custom_prompts_text)
    prompts_file = f.name

print(f"\n=== Custom Prompts (from {prompts_file}) ===")
prompts = load_prompts(prompts_file, 5)
for i, p in enumerate(prompts, 1):
    print(f"  {i}. {p}")

# --- Dry-run benchmark with custom prompts ---
print("\n=== Dry-Run Benchmark Results ===")
base = {
    "perplexity": 18.4321,
    "tokens_per_sec": 42.7,
    "latency_mean_ms": 23.42,
    "latency_p95_ms": 30.15,
    "latency_std_ms": 2.11,
}
variants = {"float16": {**base, "size_ratio": 1.0, "estimated": False}}
for qt in ("Q4_K_M", "Q8_0"):
    variants[qt] = estimate_quantized_metrics(base, qt)

for variant, res in variants.items():
    print(
        f"  {variant:<10}  PPL={res['perplexity']:.4f}  "
        f"speed={res['tokens_per_sec']:.1f} tok/s  "
        f"lat_mean={res.get('latency_mean_ms', 0):.2f} ms"
    )

# --- Memory table for target model ---
print("\n=== Memory Requirements (9B model, current system) ===")
available_gb = get_available_ram_gb()
md_table = format_markdown_table(9.0, available_gb, model_id="SandyVeliz/acervo-extractor-qwen3.5-9b")
print(md_table)

# Cleanup temp file
Path(prompts_file).unlink(missing_ok=True)
