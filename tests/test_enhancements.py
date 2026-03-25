"""
tests/test_enhancements.py — pytest suite for new features added in the enhancement pass.

Covers:
  1.  memory_estimator.py — unit tests for all public functions
  2.  memory_estimator CLI — --help, --json, --params, --model
  3.  compare.py CLI — --dry-run produces JSON + Markdown + CSV
  4.  compare.py unit tests — mock_benchmark, generate_comparison_report
  5.  benchmark.py enhancements — warmup, latency stats, CSV export, prompts-file
  6.  demo.py enhancements — warmup_runs arg, CSV export, latency in JSON
  7.  quantize.py enhancements — --list-quant-types, --retry-attempts, meta timestamp
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable


def run(args: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=kwargs.pop("timeout", 120),
        cwd=str(PROJECT_ROOT),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. memory_estimator.py unit tests
# ---------------------------------------------------------------------------

class TestMemoryEstimatorUnit:
    def setup_method(self):
        sys.path.insert(0, str(PROJECT_ROOT))

    def test_estimate_memory_q4_km(self):
        from acervo_extractor_qwe.memory_estimator import estimate_memory
        result = estimate_memory(9.0, "Q4_K_M")
        assert result["params_b"] == 9.0
        assert result["quant_type"] == "Q4_K_M"
        assert result["peak_ram_gb"] > 0
        # Q4_K_M on 9B should be well under 8 GB
        assert result["peak_ram_gb"] < 10

    def test_estimate_memory_float16_larger_than_q4(self):
        from acervo_extractor_qwe.memory_estimator import estimate_memory
        f16 = estimate_memory(9.0, "float16")
        q4 = estimate_memory(9.0, "Q4_K_M")
        assert f16["peak_ram_gb"] > q4["peak_ram_gb"]

    def test_estimate_memory_q8_between_f16_and_q4(self):
        from acervo_extractor_qwe.memory_estimator import estimate_memory
        f16 = estimate_memory(9.0, "float16")
        q8 = estimate_memory(9.0, "Q8_0")
        q4 = estimate_memory(9.0, "Q4_K_M")
        assert f16["peak_ram_gb"] > q8["peak_ram_gb"] > q4["peak_ram_gb"]

    def test_get_model_params_known_gpt2(self):
        from acervo_extractor_qwe.memory_estimator import get_model_params
        params = get_model_params("openai-community/gpt2")
        assert params is not None
        assert 0.1 < params < 0.2  # ~124M

    def test_get_model_params_known_9b(self):
        from acervo_extractor_qwe.memory_estimator import get_model_params
        params = get_model_params("SandyVeliz/acervo-extractor-qwen3.5-9b")
        assert params == 9.0

    def test_get_model_params_from_name_pattern(self):
        from acervo_extractor_qwe.memory_estimator import get_model_params
        # Should parse "7B" from model name
        params = get_model_params("some-org/some-model-7B-instruct")
        assert params == 7.0

    def test_get_model_params_unknown_returns_none(self):
        from acervo_extractor_qwe.memory_estimator import get_model_params
        result = get_model_params("unknown/model-no-size-hint")
        # May return None or a value from HF hub; just assert no exception
        assert result is None or isinstance(result, float)

    def test_recommend_quant_small_ram(self):
        from acervo_extractor_qwe.memory_estimator import recommend_quant
        # 4 GB available for a 9B model — must pick very aggressive quant
        rec = recommend_quant(9.0, 4.0)
        assert rec in ("Q2_K", "Q3_K_M", "IQ1_M")

    def test_recommend_quant_large_ram(self):
        from acervo_extractor_qwe.memory_estimator import recommend_quant
        # 64 GB available — should prefer float16
        rec = recommend_quant(9.0, 64.0)
        assert rec == "float16"

    def test_recommend_quant_none_ram_returns_default(self):
        from acervo_extractor_qwe.memory_estimator import recommend_quant
        rec = recommend_quant(9.0, None)
        assert rec == "Q4_K_M"

    def test_build_memory_table_length(self):
        from acervo_extractor_qwe.memory_estimator import build_memory_table
        table = build_memory_table(7.0)
        assert len(table) >= 7  # at least float16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K

    def test_build_memory_table_monotonically_decreasing(self):
        from acervo_extractor_qwe.memory_estimator import build_memory_table
        table = build_memory_table(7.0)
        # Peak RAM should be monotonically decreasing from float32 → IQ1_M
        ram_values = [row["peak_ram_gb"] for row in table]
        assert ram_values == sorted(ram_values, reverse=True)

    def test_format_markdown_table_contains_header(self):
        from acervo_extractor_qwe.memory_estimator import format_markdown_table
        md = format_markdown_table(9.0)
        assert "Memory Estimates" in md
        assert "Q4_K_M" in md
        assert "float16" in md

    def test_format_markdown_table_recommended_marker(self):
        from acervo_extractor_qwe.memory_estimator import format_markdown_table
        md = format_markdown_table(9.0, available_ram_gb=8.0)
        assert "recommended" in md


# ---------------------------------------------------------------------------
# 2. memory_estimator CLI
# ---------------------------------------------------------------------------

class TestMemoryEstimatorCLI:
    def test_help_exits_zero(self):
        result = run([PYTHON, "memory_estimator.py", "--help"])
        assert result.returncode == 0

    def test_help_mentions_params(self):
        result = run([PYTHON, "memory_estimator.py", "--help"])
        assert "--params" in result.stdout

    def test_params_flag_produces_output(self):
        result = run([PYTHON, "memory_estimator.py", "--params", "9.0"])
        assert result.returncode == 0
        assert "Q4_K_M" in result.stdout

    def test_json_flag_produces_valid_json(self):
        result = run([PYTHON, "memory_estimator.py", "--params", "7.0", "--json"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "params_b" in data
        assert "recommended_quant" in data
        assert "estimates" in data
        assert isinstance(data["estimates"], list)

    def test_json_estimates_have_required_keys(self):
        result = run([PYTHON, "memory_estimator.py", "--params", "7.0", "--json"])
        data = json.loads(result.stdout)
        for est in data["estimates"]:
            assert "quant_type" in est
            assert "peak_ram_gb" in est
            assert "weights_gb" in est

    def test_model_flag_with_known_model(self):
        result = run([PYTHON, "memory_estimator.py", "--model", "openai-community/gpt2"])
        assert result.returncode == 0
        assert "Q4_K_M" in result.stdout

    def test_available_ram_flag_shows_recommendation(self):
        result = run([
            PYTHON, "memory_estimator.py", "--params", "9.0", "--available-ram", "16",
        ])
        assert result.returncode == 0
        assert "recommended" in result.stdout.lower()

    def test_no_args_exits_nonzero(self):
        result = run([PYTHON, "memory_estimator.py"])
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# 3. compare.py CLI — dry-run
# ---------------------------------------------------------------------------

class TestCompareCLI:
    def test_help_exits_zero(self):
        result = run([PYTHON, "compare.py", "--help"])
        assert result.returncode == 0

    def test_help_mentions_models(self):
        result = run([PYTHON, "compare.py", "--help"])
        assert "--models" in result.stdout

    def test_dry_run_exits_zero(self, tmp_path):
        result = run([
            PYTHON, "compare.py",
            "--dry-run",
            "--output", str(tmp_path / "report.md"),
            "--results-json", str(tmp_path / "results.json"),
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_dry_run_creates_report(self, tmp_path):
        report = tmp_path / "report.md"
        run([
            PYTHON, "compare.py", "--dry-run",
            "--output", str(report),
            "--results-json", str(tmp_path / "results.json"),
        ])
        assert report.exists()

    def test_dry_run_creates_json(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "compare.py", "--dry-run",
            "--output", str(tmp_path / "report.md"),
            "--results-json", str(json_file),
        ])
        assert json_file.exists()

    def test_dry_run_json_valid(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "compare.py", "--dry-run",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        data = json.loads(json_file.read_text())
        assert "models" in data
        assert "results" in data
        assert "generated_at" in data
        assert isinstance(data["results"], list)

    def test_dry_run_results_have_metrics(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "compare.py", "--dry-run",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        data = json.loads(json_file.read_text())
        for r in data["results"]:
            assert "model" in r
            assert "metrics" in r
            m = r["metrics"]
            assert "perplexity" in m
            assert "tokens_per_sec" in m

    def test_dry_run_report_contains_perplexity(self, tmp_path):
        report = tmp_path / "report.md"
        run([
            PYTHON, "compare.py", "--dry-run",
            "--output", str(report),
            "--results-json", str(tmp_path / "r.json"),
        ])
        content = report.read_text()
        assert "Perplexity" in content

    def test_dry_run_report_contains_speed(self, tmp_path):
        report = tmp_path / "report.md"
        run([
            PYTHON, "compare.py", "--dry-run",
            "--output", str(report),
            "--results-json", str(tmp_path / "r.json"),
        ])
        content = report.read_text()
        assert "Speed" in content

    def test_dry_run_export_csv(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "compare.py", "--dry-run", "--export-csv",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        csv_file = json_file.with_suffix(".csv")
        assert csv_file.exists(), "CSV file not created"
        lines = csv_file.read_text().splitlines()
        assert len(lines) >= 2  # header + at least one row
        assert "perplexity" in lines[0]

    def test_dry_run_single_model(self, tmp_path):
        result = run([
            PYTHON, "compare.py", "--dry-run",
            "--models", "openai-community/gpt2",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(tmp_path / "r.json"),
        ])
        assert result.returncode == 0

    def test_dry_run_latency_stats_in_json(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "compare.py", "--dry-run",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        data = json.loads(json_file.read_text())
        for r in data["results"]:
            m = r["metrics"]
            assert "latency_mean_ms" in m
            assert "latency_p95_ms" in m
            assert "latency_std_ms" in m


# ---------------------------------------------------------------------------
# 4. compare.py unit tests
# ---------------------------------------------------------------------------

class TestCompareUnit:
    def setup_method(self):
        sys.path.insert(0, str(PROJECT_ROOT))

    def test_mock_benchmark_deterministic(self):
        from acervo_extractor_qwe.compare import mock_benchmark
        r1 = mock_benchmark("openai-community/gpt2")
        r2 = mock_benchmark("openai-community/gpt2")
        assert r1 == r2

    def test_mock_benchmark_different_models_differ(self):
        from acervo_extractor_qwe.compare import mock_benchmark
        r1 = mock_benchmark("openai-community/gpt2")
        r2 = mock_benchmark("facebook/opt-125m")
        assert r1["perplexity"] != r2["perplexity"]

    def test_mock_benchmark_has_required_keys(self):
        from acervo_extractor_qwe.compare import mock_benchmark
        r = mock_benchmark("openai-community/gpt2")
        for key in ("perplexity", "tokens_per_sec", "latency_mean_ms",
                    "latency_p95_ms", "latency_std_ms", "measured"):
            assert key in r, f"missing key: {key}"

    def test_mock_benchmark_measured_is_false(self):
        from acervo_extractor_qwe.compare import mock_benchmark
        r = mock_benchmark("openai-community/gpt2")
        assert r["measured"] is False

    def test_generate_comparison_report_has_perplexity(self):
        from acervo_extractor_qwe.compare import generate_comparison_report
        results = [
            {"model": "model-a", "metrics": {"perplexity": 15.0, "tokens_per_sec": 40.0,
                                              "latency_mean_ms": 25.0, "latency_p95_ms": 32.0,
                                              "latency_std_ms": 2.0, "measured": False}},
            {"model": "model-b", "metrics": {"perplexity": 18.0, "tokens_per_sec": 35.0,
                                              "latency_mean_ms": 28.5, "latency_p95_ms": 37.0,
                                              "latency_std_ms": 3.0, "measured": False}},
        ]
        report = generate_comparison_report(results, {"num_prompts": 20, "max_new_tokens": 30, "warmup_runs": 2})
        assert "Perplexity" in report

    def test_generate_comparison_report_has_speed(self):
        from acervo_extractor_qwe.compare import generate_comparison_report
        results = [
            {"model": "model-a", "metrics": {"perplexity": 15.0, "tokens_per_sec": 40.0,
                                              "latency_mean_ms": 25.0, "latency_p95_ms": 32.0,
                                              "latency_std_ms": 2.0, "measured": False}},
        ]
        report = generate_comparison_report(results, {"num_prompts": 20, "max_new_tokens": 30, "warmup_runs": 2})
        assert "Speed" in report

    def test_generate_comparison_report_best_model_in_summary(self):
        from acervo_extractor_qwe.compare import generate_comparison_report
        results = [
            {"model": "fast-model", "metrics": {"perplexity": 20.0, "tokens_per_sec": 100.0,
                                                 "latency_mean_ms": 10.0, "latency_p95_ms": 13.0,
                                                 "latency_std_ms": 1.0, "measured": False}},
            {"model": "accurate-model", "metrics": {"perplexity": 12.0, "tokens_per_sec": 30.0,
                                                     "latency_mean_ms": 33.3, "latency_p95_ms": 43.0,
                                                     "latency_std_ms": 3.0, "measured": False}},
        ]
        report = generate_comparison_report(results, {"num_prompts": 20, "max_new_tokens": 30, "warmup_runs": 2})
        assert "accurate-model" in report
        assert "fast-model" in report


# ---------------------------------------------------------------------------
# 5. benchmark.py enhancements
# ---------------------------------------------------------------------------

class TestBenchmarkEnhancements:
    def test_warmup_runs_arg_accepted(self, tmp_path):
        result = run([
            PYTHON, "benchmark.py",
            "--dry-run",
            "--warmup-runs", "0",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(tmp_path / "r.json"),
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_export_csv_creates_file(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py", "--dry-run", "--export-csv",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        csv_file = json_file.with_suffix(".csv")
        assert csv_file.exists(), "CSV file not created"

    def test_export_csv_has_header(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py", "--dry-run", "--export-csv",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        csv_file = json_file.with_suffix(".csv")
        header = csv_file.read_text().splitlines()[0]
        assert "perplexity" in header
        assert "tokens_per_sec" in header

    def test_prompts_file_is_used(self, tmp_path):
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("Hello world\nWhat is AI?\nExplain Python.\n")
        result = run([
            PYTHON, "benchmark.py", "--dry-run",
            "--prompts-file", str(prompts_file),
            "--num-prompts", "3",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(tmp_path / "r.json"),
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_prompts_file_missing_exits_nonzero(self, tmp_path):
        result = run([
            PYTHON, "benchmark.py",
            "--prompts-file", str(tmp_path / "nonexistent.txt"),
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(tmp_path / "r.json"),
        ])
        assert result.returncode != 0

    def test_dry_run_json_has_generated_at(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py", "--dry-run",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        data = json.loads(json_file.read_text())
        assert "generated_at" in data
        assert "UTC" in data["generated_at"]

    def test_dry_run_json_has_warmup_runs(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py", "--dry-run", "--warmup-runs", "3",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        data = json.loads(json_file.read_text())
        assert data.get("warmup_runs") == 3

    def test_dry_run_json_has_latency_stats(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py", "--dry-run",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
        ])
        data = json.loads(json_file.read_text())
        f16 = data["variants"]["float16"]
        assert "latency_mean_ms" in f16
        assert "latency_p95_ms" in f16
        assert "latency_std_ms" in f16

    def test_report_contains_latency_column(self, tmp_path):
        report = tmp_path / "r.md"
        run([
            PYTHON, "benchmark.py", "--dry-run",
            "--output", str(report),
            "--results-json", str(tmp_path / "r.json"),
        ])
        content = report.read_text()
        assert "Lat." in content or "latency" in content.lower()

    def test_estimate_quantized_propagates_latency(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import estimate_quantized_metrics
        base = {
            "perplexity": 15.0,
            "tokens_per_sec": 50.0,
            "latency_mean_ms": 20.0,
            "latency_p95_ms": 26.0,
            "latency_std_ms": 1.5,
        }
        q4 = estimate_quantized_metrics(base, "Q4_K_M")
        # Faster → lower latency
        assert q4["latency_mean_ms"] < base["latency_mean_ms"]
        assert q4["latency_p95_ms"] < base["latency_p95_ms"]

    def test_load_prompts_from_file(self, tmp_path):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import load_prompts
        f = tmp_path / "prompts.txt"
        f.write_text("prompt one\nprompt two\nprompt three\n")
        prompts = load_prompts(str(f), 3)
        assert len(prompts) == 3
        assert prompts[0] == "prompt one"

    def test_load_prompts_uses_defaults_when_no_file(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import load_prompts
        prompts = load_prompts(None, 5)
        assert len(prompts) == 5

    def test_load_prompts_pads_to_num_prompts(self, tmp_path):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import load_prompts
        f = tmp_path / "p.txt"
        f.write_text("just one\n")
        prompts = load_prompts(str(f), 10)
        assert len(prompts) == 10

    def test_export_csv_function(self, tmp_path):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import export_csv
        results = {
            "variants": {
                "float16": {"perplexity": 15.0, "tokens_per_sec": 40.0,
                            "latency_mean_ms": 25.0, "latency_p95_ms": 32.0,
                            "latency_std_ms": 2.0, "size_ratio": 1.0,
                            "bits_per_weight": 16.0, "estimated": False},
                "Q4_K_M": {"perplexity": 15.9, "tokens_per_sec": 44.8,
                           "latency_mean_ms": 22.3, "latency_p95_ms": 28.5,
                           "latency_std_ms": 1.8, "size_ratio": 0.26,
                           "bits_per_weight": 4.5, "estimated": True},
            }
        }
        csv_path = tmp_path / "out.csv"
        export_csv(results, csv_path)
        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 3  # header + 2 variants
        assert "float16" in lines[1] or "float16" in lines[2]


# ---------------------------------------------------------------------------
# 6. demo.py enhancements
# ---------------------------------------------------------------------------

class TestDemoEnhancements:
    def test_warmup_runs_arg(self, tmp_path):
        result = run([
            PYTHON, "demo.py", "--dry-run", "--warmup-runs", "0",
            "--outputs-dir", str(tmp_path),
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_export_csv_creates_file(self, tmp_path):
        run([
            PYTHON, "demo.py", "--dry-run", "--export-csv",
            "--outputs-dir", str(tmp_path),
        ])
        assert (tmp_path / "benchmark_results.csv").exists()

    def test_export_csv_has_header(self, tmp_path):
        run([
            PYTHON, "demo.py", "--dry-run", "--export-csv",
            "--outputs-dir", str(tmp_path),
        ])
        lines = (tmp_path / "benchmark_results.csv").read_text().splitlines()
        assert "perplexity" in lines[0]

    def test_json_has_system_info(self, tmp_path):
        run([PYTHON, "demo.py", "--dry-run", "--outputs-dir", str(tmp_path)])
        data = json.loads((tmp_path / "benchmark_results.json").read_text())
        assert "system_info" in data
        assert "platform" in data["system_info"]

    def test_json_has_warmup_runs(self, tmp_path):
        run([PYTHON, "demo.py", "--dry-run", "--warmup-runs", "3",
             "--outputs-dir", str(tmp_path)])
        data = json.loads((tmp_path / "benchmark_results.json").read_text())
        assert data.get("warmup_runs") == 3

    def test_json_variants_have_latency_stats(self, tmp_path):
        run([PYTHON, "demo.py", "--dry-run", "--outputs-dir", str(tmp_path)])
        data = json.loads((tmp_path / "benchmark_results.json").read_text())
        for variant_name, variant in data["variants"].items():
            assert "latency_mean_ms" in variant, f"latency_mean_ms missing in {variant_name}"

    def test_report_contains_latency_column(self, tmp_path):
        run([PYTHON, "demo.py", "--dry-run", "--outputs-dir", str(tmp_path)])
        content = (tmp_path / "quantization_report.md").read_text()
        assert "Lat." in content or "latency" in content.lower()

    def test_build_variants_latency_propagated(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from demo import build_variants  # demo.py stays at project root
        base = {
            "perplexity": 18.0,
            "tokens_per_sec": 42.0,
            "latency_mean_ms": 23.8,
            "latency_p95_ms": 31.0,
            "latency_std_ms": 2.0,
        }
        variants = build_variants(base)
        assert "latency_mean_ms" in variants["Q4_K_M"]
        # Q4_K_M is faster → lower latency than float16
        assert variants["Q4_K_M"]["latency_mean_ms"] < base["latency_mean_ms"]


# ---------------------------------------------------------------------------
# 7. quantize.py enhancements
# ---------------------------------------------------------------------------

class TestQuantizeEnhancements:
    def test_list_quant_types_exits_zero(self):
        result = run([PYTHON, "quantize.py", "--list-quant-types"])
        assert result.returncode == 0

    def test_list_quant_types_shows_q4_km(self):
        result = run([PYTHON, "quantize.py", "--list-quant-types"])
        assert "Q4_K_M" in result.stdout

    def test_list_quant_types_shows_q8_0(self):
        result = run([PYTHON, "quantize.py", "--list-quant-types"])
        assert "Q8_0" in result.stdout

    def test_list_quant_types_does_not_run_quantization(self, tmp_path):
        result = run([
            PYTHON, "quantize.py", "--list-quant-types",
            "--output-dir", str(tmp_path),
        ])
        assert result.returncode == 0
        # No .gguf files should be created
        assert not list(tmp_path.glob("*.gguf"))

    def test_retry_attempts_arg_accepted(self, tmp_path):
        result = run([
            PYTHON, "quantize.py",
            "--dry-run",
            "--retry-attempts", "1",
            "--output-dir", str(tmp_path),
            "--model", "openai-community/gpt2",
        ])
        assert result.returncode == 0

    def test_dry_run_meta_has_generated_at(self, tmp_path):
        run([
            PYTHON, "quantize.py", "--dry-run",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
        ])
        meta = json.loads((tmp_path / "quantization_meta.json").read_text())
        assert "generated_at" in meta
        assert "UTC" in meta["generated_at"]

    def test_retry_helper_succeeds_on_first_try(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.quantize import _retry
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = _retry(fn, attempts=3, backoff=0.0, label="test")
        assert result == "ok"
        assert len(calls) == 1

    def test_retry_helper_retries_on_failure(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.quantize import _retry
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("transient")
            return "ok"

        result = _retry(fn, attempts=3, backoff=0.0, label="test")
        assert result == "ok"
        assert len(calls) == 3

    def test_retry_helper_raises_after_all_attempts(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.quantize import _retry

        def always_fail():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            _retry(always_fail, attempts=2, backoff=0.0, label="test")
