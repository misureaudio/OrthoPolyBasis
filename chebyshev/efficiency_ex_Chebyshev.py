"""
  CHEBYSHEV POLYNOMIALS - EFFICIENCY BEST PRACTICES
=================================================
"""
import sys
sys.path.insert(0, "C:\\Users\\MATTIA\\source\\repos\\OrthoPolyB")

from chebyshev import chebyshev_coefficients, get_chebyshev_zeros
from chebyshev.core import ChebyshevGenerator
from chebyshev.quadrature_roots import ChebyshevQuadrature
import numpy as np
import time

print("=" * 70)
print("SCENARIO: Evaluate T_100(x) at 5,000 points")
print("=" * 70)

n = 100
num_points = 5000
x_values = np.linspace(-1.0, 1.0, num_points)

# NAIVE: Using convenience function (creates new generator each time)
print("\n[NAIVE] Using chebyshev_coefficients() + manual evaluation...")
start = time.perf_counter()
gen_naive = ChebyshevGenerator()
for x in x_values:
    coeffs = chebyshev_coefficients(n)  # New calculation each time!
    result = gen_naive.evaluate_series(x, coeffs)
time_naive = time.perf_counter() - start
print(f"  Time: {time_naive:.4f} seconds")

# OPTIMIZED: Get coefficients ONCE, evaluate many times
print("\n[OPTIMIZED] Get coefficients once, reuse for all evaluations...")
gen = ChebyshevGenerator()
coeffs = gen.get_monomial_coefficients(n)# Calculated and cached once
start = time.perf_counter()
for x in x_values:
    result = gen.evaluate_series(x, coeffs)
time_optimized = time.perf_counter() - start
print(f"  Time: {time_optimized:.4f} seconds")

print(f"\n  SPEEDUP: {time_naive/time_optimized:.1f}x faster")