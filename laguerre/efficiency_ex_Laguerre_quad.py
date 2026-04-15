"""
LAGUERRE QUADRATURE - EFFICIENCY BEST PRACTICES
"""
import sys
sys.path.insert(0, "C:\\\\Users\\\\MATTIA\\\\source\\\\repos\\\\OrthoPolyB")

from laguerre.basis import GeneralizedLaguerreBasis
import numpy as np
import time

num_functions = 500
n_quad = 64

print("=" * 70)
print(f"SCENARIO: Integrate {num_functions} functions using n_quad={n_quad} for Gauss-Laguerre")
print("=" * 70)


# NAIVE: Create basis and compute quadrature for each integration
print("\n[NAIVE] Computing quadrature for each integration...")
start = time.perf_counter()
for i in range(num_functions):
    basis = GeneralizedLaguerreBasis(n_quad, alpha=0)  # Creates full basis
    nodes, weights = basis._gauss_quadrature(n_quad)   # Extracts quadrature
    f = lambda x, k=i: np.sin((k+1)*np.sqrt(x)) * np.exp(-x/2)
    result = np.sum(weights * f(nodes))
time_naive = time.perf_counter() - start
print(f"  Time: {time_naive:.4f} seconds")

# OPTIMIZED: Create basis once, reuse quadrature
print("\n[OPTIMIZED] Computing quadrature once...")
basis = GeneralizedLaguerreBasis(n_quad, alpha=0)
nodes, weights = basis._gauss_quadrature(n_quad)
start = time.perf_counter()
for i in range(num_functions):
    f = lambda x, k=i: np.sin((k+1)*np.sqrt(x)) * np.exp(-x/2)
    result = np.sum(weights * f(nodes))
time_optimized = time.perf_counter() - start
print(f"  Time: {time_optimized:.4f} seconds")

speedup = time_naive / time_optimized
print(f"\\n  SPEEDUP: {speedup:.1f}x faster")
