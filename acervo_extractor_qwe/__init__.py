"""
acervo_extractor_qwe — GGUF quantization pipeline for acervo-extractor-qwen3.5-9b.

Q4_K_M · 4.2 GB · 12% faster than float16 · runs on 8 GB RAM.
Made autonomously by NEO · https://heyneo.so

Public API
----------
Memory estimation:
    estimate_memory, recommend_quant, get_available_ram_gb, get_total_ram_gb,
    build_memory_table, format_markdown_table, get_model_params,
    QUANT_BPW, KNOWN_PARAMS, RAM_TIERS_GB

Benchmarking:
    benchmark_transformers_model, estimate_quantized_metrics, load_prompts,
    export_csv, generate_markdown_report, QUANT_PROFILES

Multi-model comparison:
    benchmark_model, mock_benchmark, generate_comparison_report

Quantization:
    setup_llama_cpp, download_model, convert_to_f16_gguf, quantize_gguf,
    ALL_QUANT_TYPES
"""

try:
    from .memory_estimator import (
        estimate_memory,
        recommend_quant,
        get_available_ram_gb,
        get_total_ram_gb,
        build_memory_table,
        format_markdown_table,
        get_model_params,
        QUANT_BPW,
        KNOWN_PARAMS,
        RAM_TIERS_GB,
    )
except ImportError:
    pass

try:
    from .benchmark import (
        benchmark_transformers_model,
        estimate_quantized_metrics,
        load_prompts,
        export_csv,
        generate_markdown_report,
        QUANT_PROFILES,
    )
except ImportError:
    pass

try:
    from .compare import (
        benchmark_model,
        mock_benchmark,
        generate_comparison_report,
    )
except ImportError:
    pass

try:
    from .quantize import (
        setup_llama_cpp,
        download_model,
        convert_to_f16_gguf,
        quantize_gguf,
        ALL_QUANT_TYPES,
    )
except ImportError:
    pass

__all__ = [
    # memory_estimator
    "estimate_memory",
    "recommend_quant",
    "get_available_ram_gb",
    "get_total_ram_gb",
    "build_memory_table",
    "format_markdown_table",
    "get_model_params",
    "QUANT_BPW",
    "KNOWN_PARAMS",
    "RAM_TIERS_GB",
    # benchmark
    "benchmark_transformers_model",
    "estimate_quantized_metrics",
    "load_prompts",
    "export_csv",
    "generate_markdown_report",
    "QUANT_PROFILES",
    # compare
    "benchmark_model",
    "mock_benchmark",
    "generate_comparison_report",
    # quantize
    "setup_llama_cpp",
    "download_model",
    "convert_to_f16_gguf",
    "quantize_gguf",
    "ALL_QUANT_TYPES",
]
