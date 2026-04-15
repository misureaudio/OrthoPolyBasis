"""
HERMITE POLYNOMIALS - EFFICIENCY BEST PRACTICES
===============================================
"""
import sys
sys.path.insert(0, "C:\\Users\\MATTIA\\source\\repos\\OrthoPolyB")

from hermite.numerical import HermitePolynomial, hermite_numerical_basis
import numpy as np
import time

print("=" * 70)
print("SCENARIO: Evaluate H_0 through H_50 at 1,000 points each")
print("=" * 70)

max_degree = 50
num_points = 1000
x_values = np.linspace(-3.0, 3.0, num_points)

# NAIVE: Create new polynomial object for each evaluation loop
print("\n[NAIVE] Creating HermitePolynomial objects repeatedly...")
start = time.perf_counter()
for _ in range(10):  # Simulate repeated work
    for n in range(max_degree + 1):
        H_n = HermitePolynomial(n)  # Recalculates coefficients!
        results = H_n.evaluate(x_values)
time_naive = time.perf_counter() - start
print(f"Time: {time_naive:.4f} seconds")

# OPTIMIZED: Create basis ONCE, reuse for all evaluations
print("\n[OPTIMIZED] Creating basis once, reusing for all evaluations...")
basis = hermite_numerical_basis(max_degree)# All polynomials created once
start = time.perf_counter()
for _ in range(10):
    for H_n in basis:
        results = H_n.evaluate(x_values)
time_optimized = time.perf_counter() - start
print(f"  Time: {time_optimized:.4f} seconds")

print(f"\n  SPEEDUP: {time_naive/time_optimized:.1f}x faster")