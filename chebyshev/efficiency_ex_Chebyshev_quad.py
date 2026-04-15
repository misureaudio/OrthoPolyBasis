"""
CHEBYSHEV QUADRATURE - EFFICIENCY BEST PRACTICES
"""
import sys
sys.path.insert(0, "C:\\\\Users\\\\MATTIA\\\\source\\\\repos\\\\OrthoPolyB")

from chebyshev import clenshaw_curtis_integrate, gauss_chebyshev_integrate
from chebyshev.quadrature_roots import ChebyshevQuadrature
import numpy as np
import time

num_functions = 500
n_quad = 128

print("=" * 70)
print(f"SCENARIO: Integrate {num_functions} functions using Clenshaw-Curtis n={n_quad}")
print("=" * 70)

num_functions = 500
n_quad = 128

# NAIVE: Uses convenience function that creates new object each time
print("[NAIVE] Using clenshaw_curtis_integrate() for each integration...")
start = time.perf_counter()
for i in range(num_functions):
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = clenshaw_curtis_integrate(f, n_quad)
time_naive = time.perf_counter() - start
print(f"  Time: {time_naive:.4f} seconds")

# OPTIMIZED: Create quadrature object once, reuse
print("\\n[OPTIMIZED] Creating ChebyshevQuadrature once, reusing...")
quad = ChebyshevQuadrature()
start = time.perf_counter()
for i in range(num_functions):
    f = lambda x, k=i: np.sin((k+1)*x) * np.exp(-x**2/2)
    result = quad.clenshaw_curtis_quadrature(f, n_quad)
time_optimized = time.perf_counter() - start
print(f"  Time: {time_optimized:.4f} seconds")

speedup = time_naive / time_optimized
print(f"\\n  SPEEDUP: {speedup:.1f}x faster")
