"""
Test NumPy vs mpmath paths for Legendre quadrature
"""
import sys
sys.path.insert(0, r"C:\Users\MATTIA\source\codex-dir\OrthoPolyB_np_mp")

from legendre.quadrature import LegendreQuadrature
import numpy as np
import time

print("=" * 70)
print("LEGENDRE QUADRATURE: NumPy vs mpmath Comparison")
print("=" * 70)

# Test parameters
n_quad = 64
num_functions = 500

print(f"\nn_quad={n_quad}, num_functions={num_functions}")
print()

# Test 1: mpmath path (high precision)
print("[MPMATH] High-precision path (use_mpmath=True)...")
start = time.perf_counter()
for i in range(num_functions):
    quad = LegendreQuadrature(n_quad, use_mpmath=True)
    nodes, weights = quad.nodes, quad.weights
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = np.sum(weights * f(nodes))
time_mpmath = time.perf_counter() - start
print(f"  Time: {time_mpmath:.4f} seconds")

# Test 2: NumPy path (fast, double precision)
print("\n[NUMPY] Fast path (use_mpmath=False)...")
start = time.perf_counter()
for i in range(num_functions):
    quad = LegendreQuadrature(n_quad, use_mpmath=False)
    nodes, weights = quad.nodes, quad.weights
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = np.sum(weights * f(nodes))
time_numpy = time.perf_counter() - start
print(f"  Time: {time_numpy:.4f} seconds")

# Speedup comparison
speedup = time_mpmath / time_numpy
print(f"\n  SPEEDUP (NumPy vs mpmath): {speedup:.1f}x faster")

# Accuracy comparison for a known integral
print("\n" + "=" * 70)
print("ACCURACY TEST: Integral of x^2 from -1 to +1 = 2/3")
print("=" * 70)

exact = 2.0 / 3.0
f_test = lambda x: x ** 2

quad_mp = LegendreQuadrature(128, use_mpmath=True)
result_mp = quad_mp.integrate(f_test)

quad_np = LegendreQuadrature(128, use_mpmath=False)
result_np = quad_np.integrate(f_test)

print(f"\nExact value:     {exact:.15f}")
print(f"mpmath result:   {result_mp:.15f}  (error: {abs(result_mp - exact):.2e})")
print(f"NumPy result:    {result_np:.15f}  (error: {abs(result_np - exact):.2e})")
