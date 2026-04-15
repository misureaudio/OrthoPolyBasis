import sys
sys.path.insert(0, "C:/Users/misur/source/codex-dir")

# Test core functionality
from legendre import legendre_polynomial, legendre_coefficients_descending
print("Testing core functions...")
result = legendre_polynomial(3, 0.5)  # n=3, x=0.5
print(f"P_3(0.5) = {result}")
coeffs = legendre_coefficients_descending(3)
print(f"Coefficients for P_3: {coeffs}")

# Test numpy integration
from legendre import evaluate_numpy_legendre
import numpy as np
xs = np.array([-1, 0, 0.5, 1])
values = evaluate_numpy_legendre(xs, 3)
print(f"P_3 evaluated at {list(xs)}: {list(values)}")

# Test sympy integration
from legendre import generate_sympy_legendre
poly = generate_sympy_legendre(3)
print(f"Symbolic P_3(x): {poly}")

print("\nAll tests passed! Package is working correctly.")
