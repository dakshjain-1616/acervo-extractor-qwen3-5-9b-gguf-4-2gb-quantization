#!/usr/bin/env python3
"""
04_full_pipeline.py — End-to-end workflow: plan → benchmark → report → compare.

Demonstrates the full project capability in a single runnable script:
  Step 1 — Memory planning: determine the right quant for available hardware
  Step 2 — Quantization dry-run: create stub GGUF files without downloading a model
  Step 3 — Benchmark: mock baseline + quant-tier estimation
  Step 4 — Report generation: Markdown + JSON + CSV
  Step 5 — Multi-model comparison
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, tempfile
from pathlib import Path

from acervo_extractor_qwe import (
    # Memory planning
    estimate_memory, recommend_quant, get_available_ram_gb,
    format_markdown_table, get_model_params,
    # Benchmarking
    estimate_quantized_metrics, generate_markdown_report, export_csv,
    # Multi-model comparison
    mock_benchmark, generate_comparison_report,
    # Quantization constants
    ALL_QUANT_TYPES,
)
from acervo_extractor_qwe.quantize import _create_dry_run_gguf

TARGET_MODEL = "SandyVeliz/acervo-extractor-qwen3.5-9b"
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="acervo_pipeline_"))

print(f"Pipeline output directory: {OUTPUT_DIR}\n")

# ---------------------------------------------------------------------------
# Step 1: Memory Planning
# ---------------------------------------------------------------------------
print("=" * 55)
print("Step 1: Memory Planning")
print("=" * 55)

params_b = get_model_params(TARGET_MODEL)  # 9.0 B
available_gb = get_available_ram_gb()
recommended = recommend_quant(params_b, available_gb)

print(f"Model    : {TARGET_MODEL}")
print(f"Params   : {params_b:.1f}B")
print(f"RAM avail: {available_gb:.1f} GB" if available_gb else "RAM avail: unknown")
print(f"Recommend: {recommended}\n")

mem_est = estimate_memory(params_b, recommended)
print(f"  Peak RAM for {recommended}: {mem_est['peak_ram_gb']:.2f} GB  "
      f"({mem_est['bits_per_weight']:.1f} bpw)\n")

md_table = format_markdown_table(params_b, available_gb, model_id=TARGET_MODEL)
(OUTPUT_DIR / "memory_table.md").write_text(md_table)
print(f"Memory table → {OUTPUT_DIR / 'memory_table.md'}")

# ---------------------------------------------------------------------------
# Step 2: Quantization Dry-Run (no model download)
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Step 2: Quantization Dry-Run")
print("=" * 55)

quant_dir = OUTPUT_DIR / "gguf"
quant_dir.mkdir()
quant_types = ["Q4_K_M", "Q8_0"]
created_files = []
for qt in quant_types:
    model_name = TARGET_MODEL.split("/")[-1]
    gguf_path = quant_dir / f"{model_name}-{qt}.gguf"
    _create_dry_run_gguf(gguf_path, TARGET_MODEL, qt)
    created_files.append(gguf_path)
    print(f"  Created stub: {gguf_path.name}")

# Write quantization metadata
meta = {
    "model": TARGET_MODEL,
    "quantization_types": quant_types,
    "dry_run": True,
    "output_files": [{"type": qt, "path": str(p)} for qt, p in zip(quant_types, created_files)],
}
(OUTPUT_DIR / "quantization_meta.json").write_text(json.dumps(meta, indent=2))
print(f"  Metadata → {OUTPUT_DIR / 'quantization_meta.json'}")

# ---------------------------------------------------------------------------
# Step 3: Benchmark (mock baseline)
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Step 3: Benchmark (mock baseline from gpt2 proxy)")
print("=" * 55)

# In dry-run mode use synthetic but realistic float16 metrics
base_result = {
    "perplexity": 18.4321,
    "tokens_per_sec": 42.7,
    "latency_mean_ms": 23.42,
    "latency_p95_ms": 30.15,
    "latency_std_ms": 2.11,
}
variants = {"float16": {**base_result, "size_ratio": 1.0, "bits_per_weight": 16.0, "estimated": False}}
for qt in quant_types:
    variants[qt] = estimate_quantized_metrics(base_result, qt)

print(f"  {'Variant':<10}  {'PPL':>8}  {'tok/s':>7}  {'mean ms':>8}  {'p95 ms':>7}  {'size':>5}")
print("  " + "-" * 56)
for variant, res in variants.items():
    size_str = "100%" if variant == "float16" else f"{res['size_ratio']*100:.0f}%"
    print(
        f"  {variant:<10}  {res['perplexity']:>8.4f}  "
        f"{res['tokens_per_sec']:>7.1f}  "
        f"{res.get('latency_mean_ms', 0):>8.2f}  "
        f"{res.get('latency_p95_ms', 0):>7.2f}  "
        f"{size_str:>5}"
    )

# ---------------------------------------------------------------------------
# Step 4: Generate Report (Markdown + JSON + CSV)
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Step 4: Generate Reports")
print("=" * 55)

full_results = {
    "model": TARGET_MODEL,
    "benchmark_model": "openai-community/gpt2",
    "num_prompts": 20,
    "max_new_tokens": 30,
    "warmup_runs": 2,
    "backend": "mock",
    "generated_at": "2026-03-24 12:00 UTC",
    "system_info": {"platform": sys.platform},
    "variants": variants,
}

report_path = OUTPUT_DIR / "quantization_report.md"
report_md = generate_markdown_report(full_results, TARGET_MODEL, str(report_path))
report_path.write_text(report_md)
print(f"  Markdown report → {report_path}")

json_path = OUTPUT_DIR / "benchmark_results.json"
json_path.write_text(json.dumps(full_results, indent=2))
print(f"  JSON results    → {json_path}")

csv_path = OUTPUT_DIR / "benchmark_results.csv"
export_csv(full_results, csv_path)
print(f"  CSV export      → {csv_path}")

# ---------------------------------------------------------------------------
# Step 5: Multi-Model Comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Step 5: Multi-Model Comparison")
print("=" * 55)

compare_models = ["openai-community/gpt2", "facebook/opt-125m", TARGET_MODEL]
compare_results = [{"model": m, "metrics": mock_benchmark(m)} for m in compare_models]

for r in compare_results:
    m = r["metrics"]
    print(f"  {r['model']:<50}  PPL={m['perplexity']:.4f}  {m['tokens_per_sec']:.1f} tok/s")

comparison_md = generate_comparison_report(
    compare_results,
    {"num_prompts": 20, "max_new_tokens": 30, "warmup_runs": 2},
)
comp_path = OUTPUT_DIR / "comparison_report.md"
comp_path.write_text(comparison_md)
print(f"\n  Comparison report → {comp_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Pipeline Complete — Output files:")
print("=" * 55)
for f in sorted(OUTPUT_DIR.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUTPUT_DIR)}")
