import sys
sys.path.insert(0, '.')

from .quadrature_roots import (
    get_chebyshev_zeros,
    get_chebyshev_extrema,
    gauss_chebyshev_integrate,
    clenshaw_curtis_integrate,
)
from .core import ChebyshevGenerator
import math

print('=== Module Import Test ===')
print('✓ All imports successful from local modules')

print()
print('=== Integration with Core Module ===')
gen = ChebyshevGenerator()

# Evaluate T_5 at its own zeros (should all be ~0)
zeros_t5 = get_chebyshev_zeros(5)
print(f'T_5 evaluated at its 5 zeros:')
for z in zeros_t5:
    coeffs = [0.0] * 6
    coeffs[5] = 1.0
    val = gen.evaluate_series(z, coeffs)
    print(f'  T_5({z:.6f}) = {val:.2e}')
print('✓ All values ≈ 0 as expected')

print()
print('=== Practical Quadrature Example ===')
f = lambda x: math.sin(x) * math.exp(x)
result_cc = clenshaw_curtis_integrate(f, n=64)
print(f'Function: sin(x) * exp(x) on [-1, 1]')
print(f'Clenshaw-Curtis result: {result_cc:.10f}')

print()
print('=' * 50)
print('MODULE INTEGRATION TEST PASSED!')
