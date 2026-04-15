# Tests for Chebyshev polynomial implementations
import sys
sys.path.insert(0, ".")

from .core import ChebyshevGenerator, chebyshev_derivative_stable, chebyshev_integral_stable
import math

def test_core_chebyshev():
    print("Testing core Chebyshev implementation...")
    gen = ChebyshevGenerator()
    
    # Test known values: T_n(cos(theta)) = cos(n*theta)
    test_cases = [(0, 0.5, 1.0), (1, 0.5, 0.5), (2, 0.5, -0.5), (3, 0.5, -1.0), (4, 0.5, -0.5)]
    for n, x, expected in test_cases:
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        result = gen.evaluate_series(x, coeffs)
        assert abs(result - expected) < 1e-10, f"T_{n}({x}) = {result}, expected {expected}"
        print(f"  T_{n}(0.5) = {result:.6f}")
    
    # Test derivative of T_3(x)
    deriv_coeffs = gen.get_derivative_series(3)
    result = gen.evaluate_series(0.5, deriv_coeffs)
    expected_deriv = 12 * (0.5)**2 - 3
    assert abs(result - expected_deriv) < 1e-10
    print(f"  T_3 derivative at x=0.5: {result:.6f} (analytical: {expected_deriv})")
    
    # Test batch evaluation
    xs = [0.0, 0.5, 1.0, -0.5]
    coeffs_t2 = [0.0] * 3
    coeffs_t2[2] = 1.0
    expected_results = [-1.0, -0.5, 1.0, -0.5]
    for x, r_exp in zip(xs, expected_results):
        result = gen.evaluate_series(x, coeffs_t2)
        assert abs(result - r_exp) < 1e-10
    print("  Batch evaluation test passed")
    
    # Test derivative series coefficients
    deriv_coeffs_n3 = gen.get_derivative_series(3)
    assert len(deriv_coeffs_n3) == 3
    assert abs(deriv_coeffs_n3[0] - 3.0) < 1e-10 and abs(deriv_coeffs_n3[2] - 6.0) < 1e-10
    print("  Derivative coefficients test passed")
    
    # Test integral series coefficients
    integ_coeffs = gen.get_integral_series(2)
    assert len(integ_coeffs) == 4
    assert abs(integ_coeffs[1] + 0.5) < 1e-10 and abs(integ_coeffs[3] - 1/6) < 1e-10
    print("  Integral coefficients test passed")
    
    # Test edge cases
    assert gen.get_derivative_series(0) == [0.0]
    integ_t0 = gen.get_integral_series(0)
    assert len(integ_t0) == 2 and abs(integ_t0[1] - 1.0) < 1e-10
    print("  Edge case tests passed")
    
    # Test stability at x=1
    for n in range(6):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        result_at_1 = gen.evaluate_series(1.0, coeffs)
        assert abs(result_at_1 - 1.0) < 1e-10
    print("  Stability at x=1 test passed")
    
    # Test stability at x=-1
    for n in range(6):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        result_at_neg1 = gen.evaluate_series(-1.0, coeffs)
        expected = (-1)**n
        assert abs(result_at_neg1 - expected) < 1e-10
    print("  Stability at x=-1 test passed")
    
    # Test convenience functions
    for n in range(5):
        result = chebyshev_derivative_stable(n, 0.5)
        coeffs = gen.get_derivative_series(n)
        expected = gen.evaluate_series(0.5, coeffs)
        assert abs(result - expected) < 1e-10
    print("  Convenience API tests passed")
    
    for n in range(2, 5):
        result = chebyshev_integral_stable(n, 0.5)
        coeffs = gen.get_integral_series(n)
        expected = gen.evaluate_series(0.5, coeffs)
        assert abs(result - expected) < 1e-10
    print("  Integral convenience API tests passed")
    
    print("Core Chebyshev tests PASSED!\n")


def test_numerical_stability():
    print("Testing numerical stability...")
    gen = ChebyshevGenerator()
    
    # Test at x=0: T_n(0) = 0 if n odd, (-1)^(n/2) if n even
    for n in range(1, 6):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        result = gen.evaluate_series(0.0, coeffs)
        if n % 2 == 1:
            expected = 0.0
        else:
            expected = (-1)**(n // 2)
        assert abs(result - expected) < 1e-10, f"T_{n}(0): {result} vs {expected}"
    print("  Stability at x=0 passed")
    
    # Test with a non-trivial series (sum of multiple Chebyshev terms)
    series = [1.0, 2.0, 3.0]  # 1*T_0 + 2*T_1 + 3*T_2
    result_at_half = gen.evaluate_series(0.5, series)
    expected_at_half = 1*1.0 + 2*0.5 + 3*(-0.5)  # = 1 + 1 - 1.5 = 0.5
    assert abs(result_at_half - expected_at_half) < 1e-10, f"Series evaluation: {result_at_half} vs {expected_at_half}"
    print("  Non-trivial series evaluation passed")
    
    print("Numerical stability tests PASSED!\n")


def main():
    try:
        test_core_chebyshev()
        test_numerical_stability()
        print("=" * 50)
        print("ALL TESTS PASSED!")
        return 0
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
