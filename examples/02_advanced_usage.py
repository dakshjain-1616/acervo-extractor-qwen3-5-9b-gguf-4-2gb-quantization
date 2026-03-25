#!/usr/bin/env python3
"""
02_advanced_usage.py — Multi-model comparison with latency stats, CSV export, and reports.

Demonstrates:
  • Mock-benchmarking multiple models side-by-side
  • Per-token latency statistics (mean, p95, std dev)
  • Generating a Markdown comparison report
  • Estimating quantized metrics from a float16 baseline
  • Exporting results to CSV
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from acervo_extractor_qwe import (
    mock_benchmark,
    generate_comparison_report,
    estimate_quantized_metrics,
    export_csv,
)

# --- Multi-model comparison (dry-run / mock mode) ---
models = [
    "openai-community/gpt2",
    "facebook/opt-125m",
    "SandyVeliz/acervo-extractor-qwen3.5-9b",
]

print("=== Multi-Model Benchmark (mock) ===\n")
results = []
for model_id in models:
    metrics = mock_benchmark(model_id)
    results.append({"model": model_id, "metrics": metrics})
    print(
        f"  {model_id}\n"
        f"    PPL={metrics['perplexity']:.4f}  "
        f"speed={metrics['tokens_per_sec']:.1f} tok/s  "
        f"mean_lat={metrics['latency_mean_ms']:.2f} ms  "
        f"p95={metrics['latency_p95_ms']:.2f} ms\n"
    )

# --- Markdown comparison report ---
cli_args = {"num_prompts": 20, "max_new_tokens": 30, "warmup_runs": 2}
report_md = generate_comparison_report(results, cli_args)
out_dir = Path(__file__).parent / "outputs"
out_dir.mkdir(exist_ok=True)
(out_dir / "comparison_report.md").write_text(report_md)
print(f"Comparison report saved → {out_dir / 'comparison_report.md'}")

# --- Quantization tier estimation from baseline ---
print("\n=== Quant-tier Estimation from gpt2 Baseline ===\n")
base = results[0]["metrics"]  # openai-community/gpt2
for qt in ("Q4_K_M", "Q8_0"):
    est = estimate_quantized_metrics(base, qt)
    print(
        f"  {qt:<8}  PPL={est['perplexity']:.4f}  "
        f"speed={est['tokens_per_sec']:.1f} tok/s  "
        f"size={est['size_ratio']*100:.0f}%  "
        f"lat_mean={est['latency_mean_ms']:.2f} ms"
    )

# --- CSV export ---
full_results = {
    "variants": {
        "float16": {**base, "size_ratio": 1.0, "bits_per_weight": 16.0, "estimated": False},
        **{qt: estimate_quantized_metrics(base, qt) for qt in ("Q4_K_M", "Q8_0")},
    }
}
csv_path = out_dir / "benchmark_variants.csv"
export_csv(full_results, csv_path)
print(f"\nCSV export saved → {csv_path}")
