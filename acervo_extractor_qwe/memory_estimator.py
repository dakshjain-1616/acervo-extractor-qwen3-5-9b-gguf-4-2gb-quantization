#!/usr/bin/env python3
"""
memory_estimator.py — Estimate RAM/VRAM requirements for LLM quantization.

Provides:
  • Per-quantization memory estimates given model parameter count
  • System RAM detection via psutil
  • Recommendation of the safest quant tier for available hardware
  • Markdown table formatting for embedding in benchmark reports
  • CLI for standalone use

Usage:
  python memory_estimator.py --model Qwen/Qwen3-8B
  python memory_estimator.py --params 9.0
  python memory_estimator.py --params 7.0 --available-ram 16
  python memory_estimator.py --model openai-community/gpt2 --json
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Rich — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    _RICH = True
    _console = Console()
    _err_console = Console(stderr=True)
except ImportError:
    _RICH = False
    _console = None  # type: ignore[assignment]
    _err_console = None  # type: ignore[assignment]


def _print_banner(suppress: bool = False) -> None:
    """Print the Rich startup banner with project name, version, and NEO attribution."""
    if suppress:
        return
    if _RICH:
        text = Text()
        text.append("acervo-extractor-quant", style="bold cyan")
        text.append(f"  v{VERSION}  ", style="bold white")
        text.append("memory-estimator\n", style="bold yellow")
        text.append(
            "Estimate RAM/VRAM requirements for any model + quantization type\n",
            style="white",
        )
        text.append("Made autonomously by ", style="dim")
        text.append("NEO", style="bold magenta")
        text.append(" · https://heyneo.so", style="dim")
        _err_console.print(Panel(text, border_style="cyan", expand=False))
    else:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  acervo-extractor-quant  v{VERSION}  [memory-estimator]", file=sys.stderr)
        print("  Estimate RAM/VRAM for any model + quant type", file=sys.stderr)
        print("  Made autonomously by NEO · https://heyneo.so", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)

# ---------------------------------------------------------------------------
# Bytes-per-weight for each quantization type (effective bits / 8)
# ---------------------------------------------------------------------------
QUANT_BPW: dict[str, float] = {
    "float32":  4.000,   # 32 bits
    "float16":  2.000,   # 16 bits
    "bfloat16": 2.000,   # 16 bits
    "Q8_0":     1.063,   # ~8.5 bits (header overhead)
    "Q6_K":     0.750,   # 6.0 bits
    "Q5_K_M":   0.688,   # 5.5 bits
    "Q4_K_M":   0.563,   # 4.5 bits
    "Q3_K_M":   0.438,   # 3.5 bits
    "Q2_K":     0.313,   # 2.5 bits
    "IQ1_M":    0.188,   # 1.5 bits
}

# KV-cache + activation overhead factor (conservative estimate)
_OVERHEAD = 1.20

# ---------------------------------------------------------------------------
# Known parameter counts (billions) for common models
# ---------------------------------------------------------------------------
KNOWN_PARAMS: dict[str, float] = {
    "facebook/opt-125m":                    0.125,
    "openai-community/gpt2":                0.124,
    "openai-community/gpt2-medium":         0.355,
    "openai-community/gpt2-large":          0.774,
    "openai-community/gpt2-xl":             1.557,
    "Qwen/Qwen3-0.6B":                      0.6,
    "Qwen/Qwen3-1.7B":                      1.7,
    "Qwen/Qwen3-8B":                        8.0,
    "Qwen/Qwen2.5-1.5B-Instruct":           1.5,
    "Qwen/Qwen2.5-3B-Instruct":             3.0,
    "Qwen/Qwen2.5-7B-Instruct":             7.0,
    "meta-llama/Llama-3.1-8B-Instruct":     8.0,
    "SandyVeliz/acervo-extractor-qwen3.5-9b": 9.0,
}

# Common RAM tiers to report fit/no-fit for
RAM_TIERS_GB = [4, 6, 8, 12, 16, 24, 32, 48, 64]


# ---------------------------------------------------------------------------
# Core functions (importable as a module)
# ---------------------------------------------------------------------------

def get_model_params(model_id: str) -> Optional[float]:
    """Return parameter count in billions, or None if not determinable."""
    if model_id in KNOWN_PARAMS:
        return KNOWN_PARAMS[model_id]
    # Parse from model ID string (e.g. "mistral-7B", "phi-3.8b")
    m = re.search(r"(\d+(?:\.\d+)?)[bB](?!\w)", model_id)
    if m:
        return float(m.group(1))
    # Try HuggingFace Hub API (safetensors metadata)
    try:
        from huggingface_hub import model_info as hf_model_info
        token = os.getenv("HF_TOKEN")
        info = hf_model_info(model_id, token=token)
        if hasattr(info, "safetensors") and info.safetensors:
            total = sum(
                v for v in (info.safetensors.get("total") or {}).values()
                if isinstance(v, (int, float))
            )
            if total > 0:
                return round(total / 1e9, 3)
    except Exception:
        pass
    return None


def estimate_memory(params_b: float, quant_type: str = "Q4_K_M") -> dict:
    """
    Estimate peak RAM in GB to run a model with `params_b` billion parameters
    at the given quantization type.

    Returns a dict with: params_b, quant_type, bytes_per_weight,
    weights_gb, peak_ram_gb.
    """
    bpw = QUANT_BPW.get(quant_type, 2.0)
    weights_gb = params_b * 1e9 * bpw / (1024 ** 3)
    peak_gb = weights_gb * _OVERHEAD
    return {
        "params_b": round(params_b, 3),
        "quant_type": quant_type,
        "bytes_per_weight": round(bpw, 4),
        "bits_per_weight": round(bpw * 8, 2),
        "weights_gb": round(weights_gb, 2),
        "peak_ram_gb": round(peak_gb, 2),
    }


def get_available_ram_gb() -> Optional[float]:
    """Return available system RAM in GB (requires psutil), or None."""
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1024 ** 3, 1)
    except ImportError:
        return None


def get_total_ram_gb() -> Optional[float]:
    """Return total system RAM in GB (requires psutil), or None."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        return None


def recommend_quant(params_b: float, available_ram_gb: Optional[float]) -> str:
    """
    Return the highest-quality quantization type that fits in available RAM
    with a 15% headroom buffer.  Falls back to Q4_K_M if psutil is absent.
    """
    if available_ram_gb is None:
        return "Q4_K_M"
    for qt in ("float16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ1_M"):
        est = estimate_memory(params_b, qt)
        if est["peak_ram_gb"] <= available_ram_gb * 0.85:
            return qt
    return "IQ1_M"


def build_memory_table(params_b: float) -> list[dict]:
    """Return a full estimate table across all common quant types."""
    return [
        estimate_memory(params_b, qt)
        for qt in ("float32", "float16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ1_M")
    ]


def format_markdown_table(
    params_b: float,
    available_ram_gb: Optional[float] = None,
    model_id: Optional[str] = None,
) -> str:
    """
    Return a Markdown table of memory estimates for all quant types.
    Suitable for embedding in benchmark reports.
    """
    recommended = recommend_quant(params_b, available_ram_gb)
    header = model_id or f"{params_b:.1f}B parameter model"
    lines = [
        f"### Memory Estimates — {header}",
        "",
        "| Quant Type | Bits/Weight | Weights (GB) | Peak RAM (GB) | Fits 8 GB | Fits 16 GB | Fits 32 GB |",
        "|------------|------------|-------------|--------------|-----------|-----------|-----------|",
    ]
    for row in build_memory_table(params_b):
        qt = row["quant_type"]
        bpw = row["bits_per_weight"]
        w_gb = row["weights_gb"]
        p_gb = row["peak_ram_gb"]
        fits8 = "✓" if p_gb <= 8 else "✗"
        fits16 = "✓" if p_gb <= 16 else "✗"
        fits32 = "✓" if p_gb <= 32 else "✗"
        rec = " ◄ recommended" if qt == recommended else ""
        lines.append(
            f"| `{qt}`{rec} | {bpw:.1f} | {w_gb:.2f} | {p_gb:.2f} | {fits8} | {fits16} | {fits32} |"
        )

    if available_ram_gb is not None:
        lines += [
            "",
            f"> **Available RAM detected: {available_ram_gb:.1f} GB** — "
            f"recommended quantization: **`{recommended}`**",
        ]
    else:
        lines += [
            "",
            f"> *Install `psutil` for automatic RAM detection and personalised recommendation.*",
            f"> Recommended default: **`{recommended}`**",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args and print memory estimates as Markdown or JSON."""
    parser = argparse.ArgumentParser(
        prog="memory_estimator.py",
        description="Estimate RAM/VRAM requirements for LLM quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python memory_estimator.py --params 9.0
  python memory_estimator.py --model Qwen/Qwen3-8B
  python memory_estimator.py --model SandyVeliz/acervo-extractor-qwen3.5-9b --json
  python memory_estimator.py --params 7.0 --available-ram 16
  python memory_estimator.py --version
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID (used to look up parameter count)")
    parser.add_argument("--params", type=float, default=None,
                        help="Model parameter count in billions (overrides --model lookup)")
    parser.add_argument("--available-ram", type=float, default=None,
                        help="Available RAM in GB (auto-detected via psutil if omitted)")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON instead of Markdown")
    args = parser.parse_args()

    _print_banner(suppress=args.json)

    params_b = args.params
    if params_b is None and args.model:
        params_b = get_model_params(args.model)
        if params_b is None:
            print(
                f"ERROR: Could not determine parameter count for '{args.model}'. "
                "Use --params to specify manually.",
                file=sys.stderr,
            )
            sys.exit(1)

    if params_b is None:
        parser.print_help()
        sys.exit(1)

    available_ram = args.available_ram
    if available_ram is None:
        available_ram = get_available_ram_gb()

    if args.json:
        recommended = recommend_quant(params_b, available_ram)
        print(json.dumps({
            "params_b": params_b,
            "available_ram_gb": available_ram,
            "total_ram_gb": get_total_ram_gb(),
            "recommended_quant": recommended,
            "estimates": build_memory_table(params_b),
        }, indent=2))
    else:
        print(format_markdown_table(params_b, available_ram, model_id=args.model))
        recommended = recommend_quant(params_b, available_ram)
        print(f"\nRecommended quantization: {recommended}")
        if available_ram:
            print(f"Available RAM: {available_ram:.1f} GB")
        total = get_total_ram_gb()
        if total:
            print(f"Total RAM:     {total:.1f} GB")


if __name__ == "__main__":
    main()
