"""
LAGUERRE POLYNOMIALS - EFFICIENCY BEST PRACTICES
================================================
"""
import sys
sys.path.insert(0, "C:\\Users\\MATTIA\\source\\repos\\OrthoPolyB")

from laguerre.polynomial import LaguerrePolynomial, GeneralizedLaguerrePolynomial
import numpy as np
import time

print("=" * 70)
print("SCENARIO: Evaluate L_0 through L_30 at 2,000 points each")
print("=" * 70)

max_degree = 30
num_points = 2000
x_values = np.linspace(0.0, 15.0, num_points)

# NAIVE: Create new polynomial for each evaluation
print("\n[NAIVE] Creating LaguerrePolynomial objects repeatedly...")
start = time.perf_counter()
for _ in range(5):
    for n in range(max_degree + 1):
        L_n = LaguerrePolynomial(n)  # Recalculates coefficients!
        results = L_n.evaluate(x_values)
time_naive = time.perf_counter() - start
print(f"Time: {time_naive:.4f} seconds")

# OPTIMIZED: Cache polynomials, reuse for all evaluations
print("\n[OPTIMIZED] Creating polynomials once, reusing for all evaluations...")
polynomials = [LaguerrePolynomial(n) for n in range(max_degree + 1)]
start = time.perf_counter()
for _ in range(5):
    for L_n in polynomials:
        results = L_n.evaluate(x_values)
time_optimized = time.perf_counter() - start
print(f"  Time: {time_optimized:.4f} seconds")

print(f"\n  SPEEDUP: {time_naive/time_optimized:.1f}x faster")