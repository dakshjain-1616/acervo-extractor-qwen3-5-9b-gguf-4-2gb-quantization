#!/usr/bin/env python3
"""
quantize.py — Convert a HuggingFace causal-LM to GGUF and quantize it with llama.cpp.

Supports Q4_K_M and Q8_0 (and any other llama.cpp quantization type).
Falls back to --dry-run mode when llama.cpp is not available.

New in this version:
  • --retry-attempts  — retry transient HuggingFace download errors with backoff
  • --list-quant-types — print all supported quantization types and exit
  • Richer metadata in quantization_meta.json (timestamp, sizes, model info)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

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
except ImportError:
    _RICH = False
    _console = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banner + coloured helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Print the Rich startup banner with project name, version, and NEO attribution."""
    if _RICH:
        text = Text()
        text.append("acervo-extractor-quant", style="bold cyan")
        text.append(f"  v{VERSION}  ", style="bold white")
        text.append("quantize\n", style="bold yellow")
        text.append(
            "Convert HuggingFace models to GGUF Q4_K_M / Q8_0 via llama.cpp\n",
            style="white",
        )
        text.append("Made autonomously by ", style="dim")
        text.append("NEO", style="bold magenta")
        text.append(" · https://heyneo.so", style="dim")
        _console.print(Panel(text, border_style="cyan", expand=False))
    else:
        print(f"\n{'=' * 60}")
        print(f"  acervo-extractor-quant  v{VERSION}  [quantize]")
        print("  Convert HuggingFace models to GGUF via llama.cpp")
        print("  Made autonomously by NEO · https://heyneo.so")
        print(f"{'=' * 60}\n")


def _ok(msg: str) -> None:
    """Print a green success message."""
    if _RICH:
        _console.print(f"[bold green]✓[/bold green]  {msg}")
    else:
        print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    """Print a yellow warning message."""
    if _RICH:
        _console.print(f"[bold yellow]⚠[/bold yellow]  {msg}")
    else:
        print(f"[WARN] {msg}")

# ---------------------------------------------------------------------------
# Env-var driven defaults — zero hard-coded values
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("MODEL_ID", "SandyVeliz/acervo-extractor-qwen3.5-9b")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
LLAMA_CPP_PATH = os.getenv("LLAMA_CPP_PATH", "llama.cpp")
LLAMA_CPP_REPO = os.getenv(
    "LLAMA_CPP_REPO", "https://github.com/ggerganov/llama.cpp"
)
HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", None)
DEFAULT_QUANT_TYPES = os.getenv("QUANT_TYPES", "Q4_K_M,Q8_0")
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))  # seconds, doubles each retry

# ---------------------------------------------------------------------------
# All llama.cpp quantization types with descriptions
# ---------------------------------------------------------------------------
ALL_QUANT_TYPES: dict[str, str] = {
    "Q2_K":    "2-bit  (~2.5 bpw) — smallest, lowest quality",
    "Q3_K_S":  "3-bit  (~3.0 bpw) — small",
    "Q3_K_M":  "3-bit  (~3.5 bpw) — medium quality",
    "Q3_K_L":  "3-bit  (~4.0 bpw) — large",
    "Q4_0":    "4-bit  (~4.0 bpw) — legacy",
    "Q4_K_S":  "4-bit  (~4.0 bpw) — small",
    "Q4_K_M":  "4-bit  (~4.5 bpw) — RECOMMENDED: best quality/size trade-off",
    "Q5_0":    "5-bit  (~5.0 bpw) — legacy",
    "Q5_K_S":  "5-bit  (~5.5 bpw) — small",
    "Q5_K_M":  "5-bit  (~5.5 bpw) — medium quality",
    "Q6_K":    "6-bit  (~6.0 bpw) — high quality, near-lossless",
    "Q8_0":    "8-bit  (~8.5 bpw) — near-lossless, larger than Q6_K",
    "F16":     "16-bit float — lossless, large",
    "F32":     "32-bit float — lossless, very large",
}


def _now() -> str:
    """Return the current UTC timestamp as a formatted string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry(fn, attempts: int, backoff: float, label: str):
    """
    Call *fn* up to *attempts* times, sleeping with exponential backoff on failure.
    Re-raises the last exception if all attempts fail.
    """
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.0f s …",
                label, attempt, attempts, exc, wait,
            )
            time.sleep(wait)
    raise last_exc


def setup_llama_cpp(llama_path: Path) -> Path:
    """Clone llama.cpp and build the quantize binary if needed."""
    if not llama_path.exists():
        logger.info("llama.cpp not found — cloning from %s", LLAMA_CPP_REPO)
        subprocess.run(
            ["git", "clone", "--depth=1", LLAMA_CPP_REPO, str(llama_path)],
            check=True,
        )
    else:
        logger.info("llama.cpp directory found: %s", llama_path)

    quant_bin = llama_path / "llama-quantize"
    if not quant_bin.exists():
        logger.info("Building llama-quantize …")
        subprocess.run(
            ["make", "-C", str(llama_path), "llama-quantize"],
            check=True,
        )

    if not quant_bin.exists():
        raise FileNotFoundError(
            f"llama-quantize binary not found at {quant_bin}. "
            "Build may have failed."
        )
    return llama_path


def download_model(model_id: str, retry_attempts: int = RETRY_ATTEMPTS) -> str:
    """
    Download model from HuggingFace Hub and return local path.
    Retries transient network errors with exponential backoff.
    """
    cache_dir = HF_CACHE_DIR or os.path.join(OUTPUT_DIR, "model_cache")
    logger.info("Downloading model: %s  →  %s", model_id, cache_dir)

    def _do_download() -> str:
        return snapshot_download(
            repo_id=model_id,
            token=HF_TOKEN,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )

    local_path = _retry(_do_download, retry_attempts, RETRY_BACKOFF, f"download({model_id})")
    logger.info("Model downloaded to: %s", local_path)
    return local_path


def convert_to_f16_gguf(model_path: str, output_path: Path, llama_path: Path) -> Path:
    """Convert a HuggingFace model directory to a F16 GGUF file."""
    # llama.cpp ships either convert_hf_to_gguf.py (new) or convert.py (old)
    for script_name in ("convert_hf_to_gguf.py", "convert.py"):
        script = llama_path / script_name
        if script.exists():
            break
    else:
        raise FileNotFoundError(
            "No convert script found inside llama.cpp. "
            "Please update llama.cpp to a recent version."
        )

    logger.info("Converting to F16 GGUF via %s …", script.name)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            model_path,
            "--outfile",
            str(output_path),
            "--outtype",
            "f16",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Conversion failed (exit {result.returncode}):\n{result.stderr}"
        )
    logger.info("F16 GGUF saved → %s", output_path)
    return output_path


def quantize_gguf(
    input_gguf: Path,
    output_gguf: Path,
    quant_type: str,
    llama_path: Path,
) -> Path:
    """Run llama-quantize on an existing GGUF file."""
    quant_bin = llama_path / "llama-quantize"
    logger.info("Quantizing %s  →  %s (%s) …", input_gguf.name, output_gguf.name, quant_type)
    result = subprocess.run(
        [str(quant_bin), str(input_gguf), str(output_gguf), quant_type],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Quantization to {quant_type} failed (exit {result.returncode}):\n"
            f"{result.stderr}"
        )
    size_gb = output_gguf.stat().st_size / (1024 ** 3)
    logger.info("  ✓  %s  →  %.2f GB", quant_type, size_gb)
    return output_gguf


def _create_dry_run_gguf(path: Path, model_id: str, quant_type: str) -> Path:
    """Write a tiny placeholder .gguf file for dry-run / testing purposes."""
    path.write_bytes(
        b"GGUF"  # magic bytes
        + f"\n# dry-run placeholder\n# model={model_id}\n# quant={quant_type}\n".encode()
    )
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args, set up llama.cpp, download model, and quantize to GGUF."""
    parser = argparse.ArgumentParser(
        prog="quantize.py",
        description="Quantize a HuggingFace model to GGUF using llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python quantize.py --model SandyVeliz/acervo-extractor-qwen3.5-9b
  python quantize.py --model openai-community/gpt2 --quant Q4_K_M
  python quantize.py --model SandyVeliz/acervo-extractor-qwen3.5-9b --dry-run
  python quantize.py --model Qwen/Qwen2.5-7B-Instruct --quant Q4_K_M,Q8_0
  python quantize.py --list-quant-types
  python quantize.py --version
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID to quantize (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory to save GGUF files (default: %(default)s)",
    )
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT_TYPES,
        help="Comma-separated quantization types, e.g. Q4_K_M,Q8_0 (default: %(default)s)",
    )
    parser.add_argument(
        "--llama-cpp",
        default=LLAMA_CPP_PATH,
        help="Path to llama.cpp directory (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building llama.cpp (assume binaries already exist)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate quantization without downloading models or running llama.cpp",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=RETRY_ATTEMPTS,
        help="Number of download retry attempts on transient failure (default: %(default)s)",
    )
    parser.add_argument(
        "--list-quant-types",
        action="store_true",
        help="Print all supported llama.cpp quantization types with descriptions and exit",
    )
    args = parser.parse_args()

    _print_banner()

    # -----------------------------------------------------------------------
    # --list-quant-types — print catalogue and exit
    # -----------------------------------------------------------------------
    if args.list_quant_types:
        print("\nSupported llama.cpp quantization types:\n")
        max_name = max(len(k) for k in ALL_QUANT_TYPES)
        for qt, desc in ALL_QUANT_TYPES.items():
            print(f"  {qt:<{max_name + 2}} {desc}")
        print()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_types = [q.strip() for q in args.quant.split(",") if q.strip()]
    model_name = args.model.split("/")[-1]

    # -----------------------------------------------------------------------
    # DRY RUN — create stub files so tests / CI can verify output layout
    # -----------------------------------------------------------------------
    if args.dry_run:
        logger.info("[DRY RUN] Simulating quantization for %s", args.model)
        output_files = []
        for qt in quant_types:
            gguf_path = output_dir / f"{model_name}-{qt}.gguf"
            _create_dry_run_gguf(gguf_path, args.model, qt)
            logger.info("[DRY RUN]  created placeholder: %s", gguf_path)
            output_files.append(
                {"type": qt, "path": str(gguf_path), "size_gb": 0.0, "dry_run": True}
            )

        meta = {
            "model": args.model,
            "quantization_types": quant_types,
            "dry_run": True,
            "generated_at": _now(),
            "output_files": output_files,
        }
        meta_path = output_dir / "quantization_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        _ok(f"Metadata → {meta_path}")
        _ok("Dry-run complete.")
        return

    # -----------------------------------------------------------------------
    # REAL QUANTIZATION
    # -----------------------------------------------------------------------
    try:
        llama_path = Path(args.llama_cpp)
        if not args.skip_build:
            llama_path = setup_llama_cpp(llama_path)

        model_path = download_model(args.model, retry_attempts=args.retry_attempts)

        f16_gguf = output_dir / f"{model_name}-f16.gguf"
        convert_to_f16_gguf(model_path, f16_gguf, llama_path)

        output_files = []
        for qt in quant_types:
            out_gguf = output_dir / f"{model_name}-{qt}.gguf"
            quantize_gguf(f16_gguf, out_gguf, qt, llama_path)
            size_gb = out_gguf.stat().st_size / (1024 ** 3)
            output_files.append(
                {"type": qt, "path": str(out_gguf), "size_gb": round(size_gb, 3)}
            )

        meta = {
            "model": args.model,
            "quantization_types": quant_types,
            "dry_run": False,
            "generated_at": _now(),
            "output_files": output_files,
        }
        meta_path = output_dir / "quantization_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        _ok(f"All done. Metadata → {meta_path}")

    except Exception as exc:
        logger.error("Quantization failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
