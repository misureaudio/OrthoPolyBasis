"""
HERMITE QUADRATURE - EFFICIENCY BEST PRACTICES
"""
import sys
sys.path.insert(0, "C:\\\\Users\\\\MATTIA\\\\source\\\\repos\\\\OrthoPolyB")

from hermite.integration import GaussHermiteQuadrature
import numpy as np
import time

num_functions = 5
# num_functions = 50
# num_functions = 500
n_quad = 64

print("=" * 70)
print(f"SCENARIO: Integrate {num_functions} functions using n={n_quad} Gauss-Hermite")
print("=" * 70)




# NAIVE: Create quadrature object for each integration
print("[NAIVE] Creating GaussHermiteQuadrature for each integration...")
start = time.perf_counter()
for i in range(num_functions):
    quad = GaussHermiteQuadrature(n_quad)
    nodes, weights = quad.roots, quad.weights  # Properties, not methods!
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = np.sum(weights * f(nodes))
time_naive = time.perf_counter() - start
print(f"  Time: {time_naive:.4f} seconds")

# OPTIMIZED: Create once, reuse
print("\\n[OPTIMIZED] Creating GaussHermiteQuadrature once...")
quad = GaussHermiteQuadrature(n_quad)
nodes, weights = quad.roots, quad.weights
start = time.perf_counter()
for i in range(num_functions):
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = np.sum(weights * f(nodes))
time_optimized = time.perf_counter() - start
print(f"  Time: {time_optimized:.4f} seconds")

speedup = time_naive / time_optimized
print(f"\\n  SPEEDUP: {speedup:.1f}x faster")

# resume 019d85ab-545d-7f70-b726-f27888101cf6
