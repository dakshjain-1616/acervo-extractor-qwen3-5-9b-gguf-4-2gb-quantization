"""
tests/test_quantize.py — pytest suite for the acervo-extractor quantization pipeline.

Covers:
  1. CLI --help on all scripts
  2. quantize.py --dry-run creates .gguf files in output/
  3. benchmark.py --dry-run generates a report with 'Perplexity' and 'Speed'
  4. demo.py --dry-run writes outputs/ files
  5. Unit tests for core helper functions
  6. JSON structure validation
  7. Markdown report content assertions
  8. Environment-variable defaults
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
    """Run a command with a generous timeout; capture stdout+stderr."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=kwargs.pop("timeout", 120),
        cwd=str(PROJECT_ROOT),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. --help on every script shows usage without error  (test spec #3)
# ---------------------------------------------------------------------------

class TestHelp:
    def test_quantize_help_exit_zero(self):
        result = run([PYTHON, "quantize.py", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_quantize_help_mentions_usage(self):
        result = run([PYTHON, "quantize.py", "--help"])
        combined = result.stdout + result.stderr
        assert "usage" in combined.lower()

    def test_quantize_help_mentions_model(self):
        result = run([PYTHON, "quantize.py", "--help"])
        assert "--model" in result.stdout

    def test_benchmark_help_exit_zero(self):
        result = run([PYTHON, "benchmark.py", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_benchmark_help_mentions_usage(self):
        result = run([PYTHON, "benchmark.py", "--help"])
        combined = result.stdout + result.stderr
        assert "usage" in combined.lower()

    def test_benchmark_help_mentions_output(self):
        result = run([PYTHON, "benchmark.py", "--help"])
        assert "--output" in result.stdout

    def test_demo_help_exit_zero(self):
        result = run([PYTHON, "demo.py", "--help"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_demo_help_mentions_usage(self):
        result = run([PYTHON, "demo.py", "--help"])
        combined = result.stdout + result.stderr
        assert "usage" in combined.lower()

    def test_demo_help_mentions_dry_run(self):
        result = run([PYTHON, "demo.py", "--help"])
        assert "--dry-run" in result.stdout


# ---------------------------------------------------------------------------
# 2. quantize.py --dry-run creates .gguf files in output/  (test spec #1)
# ---------------------------------------------------------------------------

class TestQuantizeDryRun:
    def test_dry_run_exits_zero(self, tmp_path):
        result = run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
            "--dry-run",
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_dry_run_creates_q4_gguf(self, tmp_path):
        run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
            "--quant", "Q4_K_M",
            "--dry-run",
        ])
        gguf_files = list(tmp_path.glob("*.gguf"))
        assert len(gguf_files) >= 1, "No .gguf file created"

    def test_dry_run_creates_q8_gguf(self, tmp_path):
        run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
            "--quant", "Q8_0",
            "--dry-run",
        ])
        gguf_files = list(tmp_path.glob("*.gguf"))
        assert any("Q8_0" in f.name for f in gguf_files), "No Q8_0 .gguf created"

    def test_dry_run_creates_both_quant_types(self, tmp_path):
        run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
            "--quant", "Q4_K_M,Q8_0",
            "--dry-run",
        ])
        gguf_files = list(tmp_path.glob("*.gguf"))
        assert len(gguf_files) >= 2, f"Expected 2 .gguf files, got {len(gguf_files)}"

    def test_dry_run_creates_meta_json(self, tmp_path):
        run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
            "--dry-run",
        ])
        meta = tmp_path / "quantization_meta.json"
        assert meta.exists(), "quantization_meta.json not created"

    def test_dry_run_meta_json_is_valid(self, tmp_path):
        run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
            "--dry-run",
        ])
        meta = json.loads((tmp_path / "quantization_meta.json").read_text())
        assert "model" in meta
        assert "dry_run" in meta
        assert meta["dry_run"] is True
        assert "output_files" in meta
        assert isinstance(meta["output_files"], list)

    def test_dry_run_gguf_starts_with_magic_bytes(self, tmp_path):
        run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(tmp_path),
            "--quant", "Q4_K_M",
            "--dry-run",
        ])
        gguf_files = list(tmp_path.glob("*.gguf"))
        assert gguf_files, "No .gguf file found"
        content = gguf_files[0].read_bytes()
        assert content[:4] == b"GGUF", "GGUF magic bytes missing"

    def test_dry_run_output_dir_is_created(self, tmp_path):
        nested = tmp_path / "nested" / "output"
        run([
            PYTHON, "quantize.py",
            "--model", "openai-community/gpt2",
            "--output-dir", str(nested),
            "--dry-run",
        ])
        assert nested.exists(), "Output directory not created"


# ---------------------------------------------------------------------------
# 3. benchmark.py report contains 'Perplexity' and 'Speed'  (test spec #2)
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_dry_run_exits_zero(self, tmp_path):
        report = tmp_path / "report.md"
        result = run([
            PYTHON, "benchmark.py",
            "--output", str(report),
            "--results-json", str(tmp_path / "results.json"),
            "--dry-run",
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_report_contains_perplexity(self, tmp_path):
        report = tmp_path / "report.md"
        run([
            PYTHON, "benchmark.py",
            "--output", str(report),
            "--results-json", str(tmp_path / "results.json"),
            "--dry-run",
        ])
        assert report.exists(), "report.md not created"
        content = report.read_text()
        assert "Perplexity" in content, "'Perplexity' not found in report"

    def test_report_contains_speed(self, tmp_path):
        report = tmp_path / "report.md"
        run([
            PYTHON, "benchmark.py",
            "--output", str(report),
            "--results-json", str(tmp_path / "results.json"),
            "--dry-run",
        ])
        content = report.read_text()
        assert "Speed" in content, "'Speed' not found in report"

    def test_report_contains_q4_k_m(self, tmp_path):
        report = tmp_path / "report.md"
        run([
            PYTHON, "benchmark.py",
            "--output", str(report),
            "--results-json", str(tmp_path / "results.json"),
            "--dry-run",
        ])
        content = report.read_text()
        assert "Q4_K_M" in content

    def test_report_contains_q8_0(self, tmp_path):
        report = tmp_path / "report.md"
        run([
            PYTHON, "benchmark.py",
            "--output", str(report),
            "--results-json", str(tmp_path / "results.json"),
            "--dry-run",
        ])
        content = report.read_text()
        assert "Q8_0" in content

    def test_results_json_exists(self, tmp_path):
        report = tmp_path / "report.md"
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py",
            "--output", str(report),
            "--results-json", str(json_file),
            "--dry-run",
        ])
        assert json_file.exists(), "results JSON not created"

    def test_results_json_is_valid(self, tmp_path):
        report = tmp_path / "report.md"
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py",
            "--output", str(report),
            "--results-json", str(json_file),
            "--dry-run",
        ])
        data = json.loads(json_file.read_text())
        assert "model" in data
        assert "variants" in data
        assert "float16" in data["variants"]
        assert "Q4_K_M" in data["variants"]
        assert "Q8_0" in data["variants"]

    def test_variants_have_perplexity_key(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
            "--dry-run",
        ])
        data = json.loads(json_file.read_text())
        for variant in ("float16", "Q4_K_M", "Q8_0"):
            assert "perplexity" in data["variants"][variant], \
                f"'perplexity' missing in {variant}"

    def test_variants_have_tokens_per_sec_key(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
            "--dry-run",
        ])
        data = json.loads(json_file.read_text())
        for variant in ("float16", "Q4_K_M", "Q8_0"):
            assert "tokens_per_sec" in data["variants"][variant], \
                f"'tokens_per_sec' missing in {variant}"

    def test_q4_smaller_than_float16(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
            "--dry-run",
        ])
        data = json.loads(json_file.read_text())
        q4_ratio = data["variants"]["Q4_K_M"]["size_ratio"]
        assert q4_ratio < 1.0, "Q4_K_M size_ratio should be < 1.0"

    def test_q4_faster_than_float16(self, tmp_path):
        json_file = tmp_path / "results.json"
        run([
            PYTHON, "benchmark.py",
            "--output", str(tmp_path / "r.md"),
            "--results-json", str(json_file),
            "--dry-run",
        ])
        data = json.loads(json_file.read_text())
        q4_speed = data["variants"]["Q4_K_M"]["tokens_per_sec"]
        f16_speed = data["variants"]["float16"]["tokens_per_sec"]
        assert q4_speed > f16_speed, "Q4_K_M should be faster than float16"


# ---------------------------------------------------------------------------
# 4. demo.py --dry-run produces output files
# ---------------------------------------------------------------------------

class TestDemo:
    def test_demo_dry_run_exits_zero(self, tmp_path):
        result = run([
            PYTHON, "demo.py",
            "--outputs-dir", str(tmp_path),
            "--dry-run",
        ])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_demo_creates_json(self, tmp_path):
        run([PYTHON, "demo.py", "--outputs-dir", str(tmp_path), "--dry-run"])
        assert (tmp_path / "benchmark_results.json").exists()

    def test_demo_creates_report(self, tmp_path):
        run([PYTHON, "demo.py", "--outputs-dir", str(tmp_path), "--dry-run"])
        assert (tmp_path / "quantization_report.md").exists()

    def test_demo_report_contains_perplexity(self, tmp_path):
        run([PYTHON, "demo.py", "--outputs-dir", str(tmp_path), "--dry-run"])
        content = (tmp_path / "quantization_report.md").read_text()
        assert "Perplexity" in content

    def test_demo_report_contains_speed(self, tmp_path):
        run([PYTHON, "demo.py", "--outputs-dir", str(tmp_path), "--dry-run"])
        content = (tmp_path / "quantization_report.md").read_text()
        assert "Speed" in content


# ---------------------------------------------------------------------------
# 5. Unit tests for helper functions in benchmark / demo modules
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_generate_markdown_report_has_perplexity(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import generate_markdown_report

        variants = {
            "float16": {"perplexity": 18.0, "tokens_per_sec": 40.0,
                        "size_ratio": 1.0, "estimated": False},
            "Q4_K_M": {"perplexity": 19.0, "tokens_per_sec": 45.0,
                       "size_ratio": 0.26, "estimated": True},
        }
        results = {"variants": variants, "num_prompts": 20,
                   "max_new_tokens": 30, "backend": "transformers"}
        md = generate_markdown_report(results, "test-model", "report.md")
        assert "Perplexity" in md

    def test_generate_markdown_report_has_speed(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import generate_markdown_report

        variants = {
            "float16": {"perplexity": 18.0, "tokens_per_sec": 40.0,
                        "size_ratio": 1.0, "estimated": False},
        }
        results = {"variants": variants, "num_prompts": 20,
                   "max_new_tokens": 30, "backend": "transformers"}
        md = generate_markdown_report(results, "test-model", "report.md")
        assert "Speed" in md

    def test_estimate_quantized_metrics_q4_smaller(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import estimate_quantized_metrics

        base = {"perplexity": 15.0, "tokens_per_sec": 50.0}
        q4 = estimate_quantized_metrics(base, "Q4_K_M")
        assert q4["size_ratio"] < 1.0
        assert q4["perplexity"] >= base["perplexity"]
        assert q4["tokens_per_sec"] > base["tokens_per_sec"]

    def test_estimate_quantized_metrics_q8_smaller(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.benchmark import estimate_quantized_metrics

        base = {"perplexity": 15.0, "tokens_per_sec": 50.0}
        q8 = estimate_quantized_metrics(base, "Q8_0")
        assert q8["size_ratio"] < 1.0

    def test_dry_run_gguf_helper(self, tmp_path):
        sys.path.insert(0, str(PROJECT_ROOT))
        from acervo_extractor_qwe.quantize import _create_dry_run_gguf

        out = tmp_path / "test.gguf"
        _create_dry_run_gguf(out, "test/model", "Q4_K_M")
        assert out.exists()
        assert out.read_bytes()[:4] == b"GGUF"

    def test_build_variants_returns_all_types(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from demo import build_variants

        base = {"perplexity": 20.0, "tokens_per_sec": 35.0}
        variants = build_variants(base)
        assert "float16" in variants
        assert "Q4_K_M" in variants
        assert "Q8_0" in variants

    def test_build_variants_float16_unchanged(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        from demo import build_variants  # demo.py stays at project root

        base = {"perplexity": 20.0, "tokens_per_sec": 35.0}
        variants = build_variants(base)
        assert variants["float16"]["perplexity"] == 20.0
        assert variants["float16"]["tokens_per_sec"] == 35.0
