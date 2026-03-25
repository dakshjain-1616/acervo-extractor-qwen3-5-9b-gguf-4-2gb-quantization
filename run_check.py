#!/usr/bin/env python3
"""Wrapper: checks packages, runs demo dry-run, runs pytest, collects results."""
import sys
import os
import subprocess
import json
from pathlib import Path

PROJECT = Path(__file__).parent
results = {}

# Step 1: Check packages
pkg_status = {}
for pkg in ['torch', 'transformers', 'matplotlib', 'huggingface_hub']:
    try:
        mod = __import__(pkg)
        pkg_status[pkg] = getattr(mod, '__version__', 'installed')
    except ImportError:
        pkg_status[pkg] = 'MISSING'

results['packages'] = pkg_status
print("=== PACKAGE STATUS ===")
for k, v in pkg_status.items():
    print(f"  {k}: {v}")

# Step 2: Install missing packages
to_install = {
    'transformers': 'transformers>=4.45.0',
    'matplotlib': 'matplotlib>=3.7.0',
    'huggingface_hub': 'huggingface-hub>=0.25.0',
    'torch': 'torch>=2.1.0',
}
installed_new = []
for pkg, pip_name in to_install.items():
    if pkg_status.get(pkg) == 'MISSING':
        print(f"\n=== INSTALLING {pip_name} ===")
        r = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', pip_name],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"  Successfully installed {pip_name}")
            installed_new.append(pip_name)
        else:
            print(f"  FAILED to install {pip_name}: {r.stderr[-500:]}")
results['newly_installed'] = installed_new

# Step 3: Run demo --dry-run
print("\n=== RUNNING demo.py --dry-run ===")
outputs_dir = PROJECT / 'outputs'
outputs_dir.mkdir(parents=True, exist_ok=True)
demo_r = subprocess.run(
    [sys.executable, str(PROJECT / 'demo.py'), '--dry-run', '--outputs-dir', str(outputs_dir)],
    capture_output=True, text=True, cwd=str(PROJECT)
)
results['demo_returncode'] = demo_r.returncode
results['demo_stdout'] = demo_r.stdout
results['demo_stderr'] = demo_r.stderr
print(f"  Return code: {demo_r.returncode}")
print("  STDOUT:")
print(demo_r.stdout)
if demo_r.stderr:
    print("  STDERR:")
    print(demo_r.stderr[-1000:])

# Step 4: Run pytest
print("\n=== RUNNING pytest tests/ ===")
pytest_r = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-v', '--timeout=60'],
    capture_output=True, text=True, cwd=str(PROJECT)
)
results['pytest_returncode'] = pytest_r.returncode
results['pytest_output'] = pytest_r.stdout + pytest_r.stderr
print(pytest_r.stdout[:5000])
if pytest_r.stderr:
    print("STDERR:", pytest_r.stderr[:2000])

# Step 5: List outputs/
print("\n=== OUTPUT FILES ===")
if outputs_dir.exists():
    files = list(outputs_dir.iterdir())
    results['output_files'] = [f.name for f in files]
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
else:
    print("  outputs/ directory does not exist!")
    results['output_files'] = []

# Step 6: Show quantization_report.md
print("\n=== outputs/quantization_report.md ===")
report_path = outputs_dir / 'quantization_report.md'
if report_path.exists():
    content = report_path.read_text()
    results['quantization_report'] = content
    print(content)
else:
    print("  File not found!")
    results['quantization_report'] = None

# Step 7: Show benchmark_results.json
print("\n=== outputs/benchmark_results.json ===")
json_path = outputs_dir / 'benchmark_results.json'
if json_path.exists():
    content = json_path.read_text()
    results['benchmark_json'] = content
    print(content)
else:
    print("  File not found!")
    results['benchmark_json'] = None

print("\n=== ALL DONE ===")
