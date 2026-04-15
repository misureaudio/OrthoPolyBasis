"""
Comprehensive Test Suite for Legendre Module Fixes
===================================================
This test suite verifies:
1. Coefficient ordering consistency across all modules
2. Input validation (negative n, non-integer inputs, edge cases)
3. Cross-module numerical consistency
4. API contract compliance

Run before fixes: Some tests will FAIL (expected - documents broken behavior)
Run after fixes: All tests should PASS
"""

import unittest
import math
import sys
from typing import List

class TestCoefficientOrdering(unittest.TestCase):
    """Test coefficient ordering consistency across modules."""
    
    def test_core_coefficient_order_documented(self):
        """Core module: coefficients in descending order (highest degree first)."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        # P_2(x) = (3x^2 - 1)/2 = 1.5*x^2 + 0*x - 0.5
        coeffs = gen.get_coefficients(2)
        self.assertEqual(len(coeffs), 3)
        self.assertAlmostEqual(coeffs[0], 1.5, places=10)   # x^2 term
        self.assertAlmostEqual(coeffs[1], 0.0, places=10)   # x term  
        self.assertAlmostEqual(coeffs[2], -0.5, places=10)  # constant
    
    def test_numpy_coefficient_order_ascending(self):
        """NumPy module: coefficients in ascending order (constant first)."""
        from legendre.numpy_integration import NumpyLegendreGenerator
        gen = NumpyLegendreGenerator()
        
        # P_2(x) = (3x^2 - 1)/2 = -0.5 + 0*x + 1.5*x^2
        coeffs = gen.get_coefficients(2)
        self.assertEqual(len(coeffs), 3)
        self.assertAlmostEqual(float(coeffs[0]), -0.5, places=10)   # constant
        self.assertAlmostEqual(float(coeffs[1]), 0.0, places=10)    # x term
        self.assertAlmostEqual(float(coeffs[2]), 1.5, places=10)    # x^2 term
    
    def test_coefficient_ordering_consistency_across_modules(self):
        """
        CRITICAL TEST: Verify coefficient ordering is consistent or properly converted.
        This test will FAIL before fix due to inconsistent ordering between core and numpy.
        """
        from legendre.core import LegendreGenerator
        from legendre.numpy_integration import NumpyLegendreGenerator, generate_numpy_legendre
        
        core_gen = LegendreGenerator()
        numpy_gen = NumpyLegendreGenerator()
        
        for n in range(6):
            core_coeffs = core_gen.get_coefficients(n)
            numpy_coeffs_asc = numpy_gen.get_coefficients(n)  # ascending order
            numpy_coeffs_desc = numpy_gen.get_coefficients_descending(n)  # descending order
            
            # Core uses descending order, so compare with numpy descending
            self.assertEqual(len(core_coeffs), len(numpy_coeffs_desc))
            for i, (c_core, c_numpy) in enumerate(zip(core_coeffs, numpy_coeffs_desc)):
                with self.subTest(n=n, idx=i):
                    self.assertAlmostEqual(c_core, float(c_numpy), places=10,
                        msg=f"P_{n}: Core[{i}]={c_core} != NumPy_desc[{i}]={c_numpy}")
    
    def test_convenience_functions_use_consistent_ordering(self):
        """
        Test that convenience functions return coefficients in documented order.
        legendre_coefficients() should match core module (descending).
        generate_numpy_legendre() uses numpy convention (ascending).
        """
        from legendre import legendre_coefficients, generate_numpy_legendre
        
        # P_3(x) = (5x^3 - 3x)/2 = 2.5*x^3 + 0*x^2 - 1.5*x + 0
        core_coeffs = legendre_coefficients(3)
        numpy_coeffs = generate_numpy_legendre(3)
        
        # Core: descending [2.5, 0.0, -1.5, 0.0]
        self.assertAlmostEqual(core_coeffs[0], 2.5, places=10)  # x^3
        self.assertAlmostEqual(core_coeffs[-1], 0.0, places=10) # constant
        
        # NumPy: ascending [0.0, -1.5, 0.0, 2.5]
        self.assertAlmostEqual(float(numpy_coeffs[0]), 0.0, places=10)   # constant
        self.assertAlmostEqual(float(numpy_coeffs[-1]), 2.5, places=10)  # x^3


class TestInputValidation(unittest.TestCase):
    """Test input validation - these will FAIL before fix (no validation exists)."""
    
    def test_core_negative_n_raises_error(self):
        """Core module should reject negative n with clear error."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        with self.assertRaises((ValueError, TypeError)) as ctx:
            gen.get_coefficients(-1)
        self.assertIn("negative", str(ctx.exception).lower() or "")
    
    def test_core_negative_n_evaluate_raises_error(self):
        """Core evaluate should reject negative n."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        with self.assertRaises((ValueError, TypeError)):
            gen.evaluate(0.5, -1)
    
    def test_core_non_integer_n_raises_error(self):
        """Core module should reject non-integer n."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        with self.assertRaises((ValueError, TypeError)):
            gen.get_coefficients(2.5)
    
    def test_core_string_n_raises_error(self):
        """Core module should reject string n."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        with self.assertRaises((ValueError, TypeError)):
            gen.get_coefficients("3")
    
    def test_numpy_negative_n_raises_error(self):
        """NumPy module should reject negative n."""
        from legendre.numpy_integration import NumpyLegendreGenerator
        gen = NumpyLegendreGenerator()
        
        with self.assertRaises((ValueError, TypeError)):
            gen.get_coefficients(-1)
    
    def test_sympy_negative_n_raises_error(self):
        """SymPy module should reject negative n."""
        from legendre.sympy_integration import SympyLegendreGenerator
        gen = SympyLegendreGenerator()
        
        with self.assertRaises((ValueError, TypeError)):
            gen.get_polynomial(-1)
    
    def test_mpmath_negative_n_raises_error(self):
        """Mpmath module should reject negative n."""
        from legendre.mpmath_integration import MpmathLegendreGenerator
        gen = MpmathLegendreGenerator()
        
        with self.assertRaises((ValueError, TypeError)):
            gen.evaluate("0.5", -1)
    
    def test_convenience_function_negative_n(self):
        """Convenience functions should validate input."""
        from legendre import legendre_coefficients
        
        with self.assertRaises((ValueError, TypeError)):
            legendre_coefficients(-1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_zero_polynomial(self):
        """P_0(x) = 1 for all x."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        for x in [-100, -1, 0, 0.5, 1, 100]:
            self.assertAlmostEqual(gen.evaluate(x, 0), 1.0, places=10)
    
    def test_large_n_computation(self):
        """Test computation for larger n values (performance and stability)."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        # Should complete without error or overflow
        coeffs = gen.get_coefficients(20)
        self.assertEqual(len(coeffs), 21)
        
        # P_n(1) should always equal 1
        self.assertAlmostEqual(gen.evaluate(1.0, 20), 1.0, places=5)
    
    def test_extreme_x_values(self):
        """Test evaluation at extreme x values."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        for n in range(5):
            # Should not crash on large x
            result = gen.evaluate(1e10, n)
            self.assertIsNotNone(result)
    
    def test_boundary_x_values(self):
        """Test P_n(1) = 1 and P_n(-1) = (-1)^n."""
        from legendre.core import LegendreGenerator
        gen = LegendreGenerator()
        
        for n in range(10):
            self.assertAlmostEqual(gen.evaluate(1.0, n), 1.0, places=10,
                msg=f"P_{n}(1) should equal 1")
            expected_neg1 = (-1) ** n
            self.assertAlmostEqual(gen.evaluate(-1.0, n), expected_neg1, places=10,
                msg=f"P_{n}(-1) should equal {expected_neg1}")


class TestCrossModuleConsistency(unittest.TestCase):
    """Test that all modules produce consistent numerical results."""
    
    def test_all_modules_agree_on_evaluation(self):
        """All implementations should give same result for P_n(x)."""
        from legendre.core import LegendreGenerator
        from legendre.numpy_integration import evaluate_numpy_legendre
        from legendre.sympy_integration import generate_sympy_legendre
        from legendre.mpmath_integration import evaluate_mpmath_legendre
        
        core_gen = LegendreGenerator()
        
        test_cases = [
            (0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5),
            (2, -0.5), (3, -0.5), (4, -0.5),
            (5, 0.0), (5, 1.0), (5, -1.0)
        ]
        
        for n, x in test_cases:
            core_result = core_gen.evaluate(x, n)
            numpy_result = evaluate_numpy_legendre(x, n)
            sympy_poly = generate_sympy_legendre(n)
            sympy_result = float(sympy_poly.subs("x", x))
            mpmath_result = float(evaluate_mpmath_legendre(str(x), n))
            
            with self.subTest(n=n, x=x):
                self.assertAlmostEqual(core_result, numpy_result, places=10,
                    msg=f"Core vs NumPy mismatch for P_{n}({x})")
                self.assertAlmostEqual(core_result, sympy_result, places=10,
                    msg=f"Core vs SymPy mismatch for P_{n}({x})")
                self.assertAlmostEqual(core_result, mpmath_result, places=10,
                    msg=f"Core vs Mpmath mismatch for P_{n}({x})")
    
    def test_derivative_consistency(self):
        """Derivative coefficients should be consistent."""
        from legendre.core import LegendreGenerator
        from legendre.sympy_integration import SympyLegendreGenerator
        
        core_gen = LegendreGenerator()
        sympy_gen = SympyLegendreGenerator()
        
        # P_2(x) = (3x^2 - 1)/2, d/dx = 3x
        for n in range(1, 5):
            core_deriv = core_gen.derivative_coefficients(n)
            sympy_deriv = sympy_gen.get_derivative(n)
            
            # Evaluate derivative at x=0.5 using core coefficients
            x = 0.5
            degree = len(core_deriv) - 1
            core_eval = sum(c * (x ** (degree - i)) for i, c in enumerate(core_deriv))
            sympy_eval = float(sympy_deriv.subs("x", x))
            
            with self.subTest(n=n):
                self.assertAlmostEqual(core_eval, sympy_eval, places=8,
                    msg=f"Derivative mismatch for P_{n}'({x})")


class TestAPICorrectness(unittest.TestCase):
    """Test that public API works as documented."""
    
    def test_legendre_polynomial_function_exists(self):
        """legendre_polynomial should be a callable function, not None."""
        from legendre.core import legendre_polynomial
        from legendre import legendre_polynomial as lp_imported
        
        self.assertIsNotNone(legendre_polynomial)
        self.assertTrue(callable(legendre_polynomial))
        self.assertIsNotNone(lp_imported)
        self.assertTrue(callable(lp_imported))
    
    def test_legendre_polynomial_correct_results(self):
        """legendre_polynomial(n, x) should return correct values."""
        from legendre import legendre_polynomial
        
        # Known values
        test_cases = [
            (0, 0.5, 1.0),
            (1, 0.5, 0.5),
            (2, 0.5, -0.125),      # (3*0.25 - 1)/2
            (3, 0.5, -0.4375),     # (5*0.125 - 3*0.5)/2
        ]
        
        for n, x, expected in test_cases:
            result = legendre_polynomial(n, x)
            with self.subTest(n=n, x=x):
                self.assertAlmostEqual(result, expected, places=10,
                    msg=f"P_{n}({x}) = {result}, expected {expected}")
    
    def test_all_exports_available(self):
        """All documented exports should be available from package."""
        import legendre
        
        required_exports = [
            "legendre_polynomial",
            "legendre_coefficients", 
            "legendre_derivative",
            "legendre_integral",
            "generate_sympy_legendre",
            "get_sympy_legendre_basis",
            "generate_numpy_legendre",
            "get_numpy_legendre_basis",
            "generate_mpmath_legendre",
            "get_mpmath_legendre_basis",
        ]
        
        for export in required_exports:
            with self.subTest(export=export):
                self.assertTrue(hasattr(legendre, export),
                    f"Missing export: {export}")
                self.assertIsNotNone(getattr(legendre, export))


if __name__ == "__main__":
    # Run with verbosity to see which tests pass/fail
    unittest.main(verbosity=2)
