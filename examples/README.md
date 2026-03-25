# Examples

Runnable examples for the `acervo_extractor_qwe` package.
Each script works from any directory — no installation required beyond the project's `requirements.txt`.

```bash
cd examples/
python 01_quick_start.py
python 02_advanced_usage.py
python 03_custom_config.py
python 04_full_pipeline.py
```

---

## Scripts

| Script | What it demonstrates |
|--------|---------------------|
| [`01_quick_start.py`](01_quick_start.py) | Minimal example (~20 lines): estimate peak RAM for a 9B model, auto-detect available system RAM, get a hardware-appropriate quant recommendation |
| [`02_advanced_usage.py`](02_advanced_usage.py) | Mock-benchmark multiple models side-by-side; per-token latency stats (mean, p95, std dev); Markdown comparison report; quant-tier estimation; CSV export |
| [`03_custom_config.py`](03_custom_config.py) | Override defaults with environment variables; load custom prompts from a file; memory requirements table; dry-run benchmark with custom prompt set |
| [`04_full_pipeline.py`](04_full_pipeline.py) | Full end-to-end workflow: memory planning → quantization dry-run (stub GGUF files) → mock benchmark → Markdown + JSON + CSV reports → multi-model comparison |

---

## How imports work

Every script begins with:

```python
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This adds the project root to `sys.path` so the `acervo_extractor_qwe` package is importable from any working directory without installing the package.

---

## Key package imports

```python
from acervo_extractor_qwe import (
    estimate_memory, recommend_quant, get_available_ram_gb,   # memory planning
    estimate_quantized_metrics, generate_markdown_report,      # benchmarking
    mock_benchmark, generate_comparison_report,                # comparison
    export_csv,                                                # CSV export
    ALL_QUANT_TYPES,                                           # quant catalogue
)
```

Individual submodules are also importable directly:

```python
from acervo_extractor_qwe.memory_estimator import format_markdown_table
from acervo_extractor_qwe.benchmark import load_prompts, export_csv
from acervo_extractor_qwe.compare import mock_benchmark
from acervo_extractor_qwe.quantize import ALL_QUANT_TYPES
```
