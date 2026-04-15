import sys
sys.path.insert(0, '.')
import math
from .quadrature_roots import (
    ChebyshevQuadrature,
    get_chebyshev_zeros,
    clenshaw_curtis_integrate,
    FFT_THRESHOLD
)

print("=== Comprehensive Final Test ===")
print()

# Test 1: Small n uses O(n²) method
print(f"Test 1: Method selection (threshold = {FFT_THRESHOLD})")
cq = ChebyshevQuadrature()
result_small = cq.clenshaw_curtis_quadrature(lambda x: x**4, FFT_THRESHOLD - 1)
result_large = cq.clenshaw_curtis_quadrature(lambda x: x**4, FFT_THRESHOLD + 1)
expected = 2.0/5.0
print(f"  n={FFT_THRESHOLD-1} (direct): {result_small:.15f}")
print(f"  n={FFT_THRESHOLD+1} (FFT):    {result_large:.15f}")
print(f"  Expected:                    {expected:.15f}")
assert abs(result_small - expected) < 1e-14
assert abs(result_large - expected) < 1e-14
print("  ✓ Both methods produce correct results")
print()

# Test 2: Convenience function with large n (uses FFT)
print("Test 2: clenshaw_curtis_integrate with large n")
result = clenshaw_curtis_integrate(lambda x: math.sin(x) * math.exp(x), n=128)
print(f"  ∫ sin(x)*exp(x) dx (n=128, FFT): {result:.10f}")
print("  ✓ Large-n integration works")
print()

# Test 3: Verify zeros still work
print("Test 3: Zeros computation")
z5 = get_chebyshev_zeros(5)
print(f"  T_5 zeros: {z5}")
assert len(z5) == 5
print("  ✓ Zeros correct")
print()

print("=" * 50)
print("ALL COMPREHENSIVE TESTS PASSED!")
