import math
from .quadrature_roots import (
    ChebyshevQuadrature, 
    get_chebyshev_zeros, 
    get_chebyshev_extrema,
    gauss_chebyshev_integrate,
    clenshaw_curtis_integrate
)

print('=== Testing Zeros ===')
z4 = get_chebyshev_zeros(4)
print(f'T_4 zeros: {z4}')
expected_z4 = [-0.9238795325112864, -0.38268343236508984, 0.38268343236509017, 0.9238795325112872]
print(f'Expected:   {expected_z4}')
assert all(abs(a-b) < 1e-10 for a,b in zip(z4, expected_z4)), 'Zeros mismatch!'
print('✓ Zeros test PASSED')

print()
print('=== Testing Extrema ===')
e3 = get_chebyshev_extrema(3)
print(f'T_3 extrema: {e3}')
expected_e3 = [1.0, 0.5, -0.5, -1.0]
assert all(abs(a-b) < 1e-10 for a,b in zip(e3, expected_e3)), 'Extrema mismatch!'
print('✓ Extrema test PASSED')

print()
print('=== Testing Gauss-Chebyshev Quadrature ===')
# ∫_{-1}^{1} 1/√(1-x²) dx = π
result = gauss_chebyshev_integrate(lambda x: 1.0, n=64)
print(f'∫ 1/√(1-x²) dx ≈ {result:.10f}, expected π = {math.pi:.10f}')
assert abs(result - math.pi) < 1e-10, f'Gauss-Chebyshev failed: {result} vs {math.pi}'
print('✓ Gauss-Chebyshev test PASSED')

# ∫_{-1}^{1} x²/√(1-x²) dx = π/2
result2 = gauss_chebyshev_integrate(lambda x: x**2, n=64)
print(f'∫ x²/√(1-x²) dx ≈ {result2:.10f}, expected π/2 = {math.pi/2:.10f}')
assert abs(result2 - math.pi/2) < 1e-10, f'Gauss-Chebyshev x² failed'
print('✓ Gauss-Chebyshev x² test PASSED')

print()
print('=== Testing Clenshaw-Curtis Quadrature ===')
# ∫_{-1}^{1} e^x dx = e - 1/e
result3 = clenshaw_curtis_integrate(math.exp, n=32)
expected3 = math.e - 1/math.e
print(f'∫ e^x dx ≈ {result3:.10f}, expected e-1/e = {expected3:.10f}')
assert abs(result3 - expected3) < 1e-6, f'Clenshaw-Curtis failed'
print('✓ Clenshaw-Curtis test PASSED')

# ∫_{-1}^{1} x² dx = 2/3
result4 = clenshaw_curtis_integrate(lambda x: x**2, n=32)
expected4 = 2.0/3.0
print(f'∫ x² dx ≈ {result4:.10f}, expected 2/3 = {expected4:.10f}')
assert abs(result4 - expected4) < 1e-6, f'Clenshaw-Curtis x² failed'
print('✓ Clenshaw-Curtis x² test PASSED')

print()
print('=' * 50)
print('ALL QUADRATURE TESTS PASSED!')
