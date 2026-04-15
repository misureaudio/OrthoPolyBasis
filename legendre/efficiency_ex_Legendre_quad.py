"""
LEGENDRE QUADRATURE - EFFICIENCY BEST PRACTICES
"""
import sys
sys.path.insert(0, "C:\\\\Users\\\\MATTIA\\\\source\\\\repos\\\\OrthoPolyB")

from legendre import gauss_legendre
import numpy as np
import time

n_quad = 128
num_functions = 500

print("=" * 70)
print(f"SCENARIO: Integrate {num_functions} functions using n={n_quad} Gauss-Legendre")
print("=" * 70)
print()
print("Note: Golub-Welsch uses eigenvalue decomposition O(n³)")
print("      For n=128, this is ~2 million operations per call!")



# NAIVE approach
print("[NAIVE] Computing quadrature fresh for each integration...")
def naive_integrate(func, n):
    nodes, weights = gauss_legendre(n)
    return np.sum(weights * func(nodes))

start = time.perf_counter()
for i in range(num_functions):
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = naive_integrate(f, n_quad)
time_naive = time.perf_counter() - start
print(f"  Time: {time_naive:.4f} seconds")

# OPTIMIZED approach
print("\\n[OPTIMIZED] Computing quadrature once, reusing for all...")
nodes, weights = gauss_legendre(n_quad)
start = time.perf_counter()
for i in range(num_functions):
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = np.sum(weights * f(nodes))
time_optimized = time.perf_counter() - start
print(f"  Time: {time_optimized:.4f} seconds")

speedup = time_naive / time_optimized
print(f"\\n  SPEEDUP: {speedup:.1f}x faster")
print(f"  Eigenvalue decompositions saved: {num_functions - 1}")
