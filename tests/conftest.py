"""Force single-threaded numerics during tests.

Why: PyTorch + LightGBM + sklearn each default to oversubscribing all CPU
cores. On macOS arm64 they deadlock against each other (4-5 minute hangs
on `pytest` with the ML extra). Setting these BEFORE numpy is imported
fixes it; conftest.py runs first.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
