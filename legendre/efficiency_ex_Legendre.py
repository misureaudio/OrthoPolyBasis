"""
LEGENDRE POLYNOMIALS - EFFICIENCY BEST PRACTICES
"""
import sys
sys.path.insert(0, "C:\\\\Users\\\\MATTIA\\\\source\\\\repos\\\\OrthoPolyB")

from legendre import legendre_polynomial, gauss_legendre
from legendre.core import LegendreGenerator
import numpy as np
import time

# SCENARIO 1: Evaluating P_n(x) at many points
print("SCENARIO 1: Evaluate P_50(x) at 10,000 equidistant points")
n, num_points = 50, 10000
x_values = np.linspace(-1.0, 1.0, num_points)

# NAIVE: Creates new generator for EACH evaluation
start = time.perf_counter()
results_naive = [legendre_polynomial(n, x) for x in x_values]
time_naive = time.perf_counter() - start
print(f"  NAIVE: {time_naive:.4f} seconds")

# OPTIMIZED: Reuse single generator instance
gen = LegendreGenerator()
start = time.perf_counter()
results_optimized = [gen.evaluate(x, n) for x in x_values]
time_optimized = time.perf_counter() - start
print(f"  OPTIMIZED: {time_optimized:.4f} seconds")
print(f"  SPEEDUP: {time_naive/time_optimized:.1f}x faster\\n")
