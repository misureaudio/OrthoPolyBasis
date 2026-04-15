"""
Test NumPy vs mpmath paths for Laguerre quadrature
"""
import sys
sys.path.insert(0, r"C:\Users\MATTIA\source\codex-dir\OrthoPolyB_np_mp")

from laguerre.basis import LaguerreQuadrature
import numpy as np
import time

print("=" * 70)
print("LAGUERRE QUADRATURE: NumPy vs mpmath Comparison")
print("=" * 70)

# Test parameters
n_quad = 64
alpha = 0.0
num_functions = 500

print(f"\nn_quad={n_quad}, alpha={alpha}, num_functions={num_functions}")
print()

# Test 1: mpmath path (high precision)
print("[MPMATH] High-precision path (use_mpmath=True)...")
start = time.perf_counter()
for i in range(num_functions):
    quad = LaguerreQuadrature(n_quad, alpha=alpha, use_mpmath=True)
    nodes, weights = quad.nodes, quad.weights
    f = lambda x, k=i: np.sin((k+1)*np.sqrt(x)) * np.exp(-x/2)
    result = np.sum(weights * f(nodes))
time_mpmath = time.perf_counter() - start
print(f"  Time: {time_mpmath:.4f} seconds")

# Test 2: NumPy path (fast, double precision)
print("\n[NUMPY] Fast path (use_mpmath=False)...")
start = time.perf_counter()
for i in range(num_functions):
    quad = LaguerreQuadrature(n_quad, alpha=alpha, use_mpmath=False)
    nodes, weights = quad.nodes, quad.weights
    f = lambda x, k=i: np.sin((k+1)*np.sqrt(x)) * np.exp(-x/2)
    result = np.sum(weights * f(nodes))
time_numpy = time.perf_counter() - start
print(f"  Time: {time_numpy:.4f} seconds")

# Speedup comparison
speedup = time_mpmath / time_numpy
print(f"\n  SPEEDUP (NumPy vs mpmath): {speedup:.1f}x faster")

# Accuracy comparison for a known integral
print("\n" + "=" * 70)
print("ACCURACY TEST: Integral of e^{-x} from 0 to 8 = 1.0")
print("=" * 70)

exact = 1.0
f_test = lambda x: np.ones_like(x)  # Integrate weight function itself

quad_mp = LaguerreQuadrature(128, alpha=alpha, use_mpmath=True)
result_mp = quad_mp.integrate(f_test)

quad_np = LaguerreQuadrature(128, alpha=alpha, use_mpmath=False)
result_np = quad_np.integrate(f_test)

print(f"\nExact value:     {exact:.15f}")
print(f"mpmath result:   {result_mp:.15f}  (error: {abs(result_mp - exact):.2e})")
print(f"NumPy result:    {result_np:.15f}  (error: {abs(result_np - exact):.2e})")
