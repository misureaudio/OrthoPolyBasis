import sys
sys.path.insert(0, "C:\\Users\\MATTIA\\source\\repos\\OrthoPolyB")

from legendre import legendre_polynomial
import numpy as np

# Definition interval for Legendre polynomials
a, b = -1.0, 1.0

# Degree of the polynomial
degree = 9

# Number of equidistant points
num_points = 25

# Generate 25 equidistant points in [-1, 1]
points = np.linspace(a, b, num_points)

# Print header
print(f"Legendre Polynomial P_{degree}(x) - {num_points} Equidistant Points")
print("=" * 60)
print(f"{'Index':<8}{'x':<18}{'P_9(x)':<20}")
print("-" * 60)

# Evaluate and print table
for i, x in enumerate(points):
    value = legendre_polynomial(degree, x)
    print(f"{i:<8}{x:<18.10f}{value:<20.15f}")

print("=" * 60)
