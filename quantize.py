#!/usr/bin/env python3
"""Thin entry-point wrapper — core implementation lives in acervo_extractor_qwe.quantize."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acervo_extractor_qwe.quantize import main

if __name__ == "__main__":
    main()
