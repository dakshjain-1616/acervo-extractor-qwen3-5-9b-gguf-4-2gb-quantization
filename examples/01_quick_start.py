#!/usr/bin/env python3
"""
01_quick_start.py — Minimal working example: memory estimation in ~15 lines.

Demonstrates:
  • Estimating peak RAM for a 9B model at Q4_K_M quantization
  • Auto-detecting available system RAM
  • Getting a hardware-appropriate quant recommendation
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acervo_extractor_qwe import estimate_memory, recommend_quant, get_available_ram_gb

# Estimate memory for the target 9B model at Q4_K_M
est = estimate_memory(9.0, "Q4_K_M")
print(f"Q4_K_M  →  {est['peak_ram_gb']:.2f} GB peak RAM  ({est['bits_per_weight']:.1f} bits/weight)")

# Auto-detect available RAM and get a recommendation
available_gb = get_available_ram_gb()
if available_gb:
    print(f"Available RAM: {available_gb:.1f} GB")
else:
    print("Available RAM: unknown (install psutil for auto-detection)")

recommended = recommend_quant(9.0, available_gb)
print(f"Recommended quant for your hardware: {recommended}")

# Compare memory across key quant tiers
print("\nQuant tier comparison for 9B model:")
for qt in ("float16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"):
    r = estimate_memory(9.0, qt)
    fits_8gb = "✓" if r["peak_ram_gb"] <= 8 else "✗"
    print(f"  {qt:<10}  {r['peak_ram_gb']:5.2f} GB  8GB: {fits_8gb}")
