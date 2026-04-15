import sys
sys.path.insert(0, "C:\\Users\\MATTIA\\source\\repos\\OrthoPolyB")

from legendre.core import LegendreGenerator
import numpy as np

# Create ONE generator instance - coefficients cached for all evaluations!
gen = LegendreGenerator()

degree = 9
num_points = 25
points = np.linspace(-1.0, 1.0, num_points)

print(f"Legendre Polynomial P_{degree}(x) - {num_points} Equidistant Points")
print("=" * 60)
print(f"{'Index':<8}{'x':<18}{'P_9(x)':<20}")
print("-" * 60)

for i, x in enumerate(points):
    value = gen.evaluate(x, degree)  # Reuses cached coefficients!
    print(f"{i:<8}{x:<18.10f}{value:<20.15f}")

print("=" * 60)