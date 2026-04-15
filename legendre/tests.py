import unittest
import math
from typing import List, Tuple

class TestLegendreCore(unittest.TestCase):
    """Test suite for the core Legendre polynomial generator."""
    
    def setUp(self):
        from legendre.core import LegendreGenerator
        self.gen = LegendreGenerator()
    
    def test_p0_polynomial(self):
        """Test P_0(x) = 1"""
        coeffs = self.gen.get_coefficients(0)
        self.assertEqual(coeffs, [1.0])
        self.assertAlmostEqual(self.gen.evaluate(5.0, 0), 1.0)
    
    def test_p1_polynomial(self):
        """Test P_1(x) = x"""
        coeffs = self.gen.get_coefficients(1)
        self.assertEqual(coeffs, [1.0, 0.0])
        self.assertAlmostEqual(self.gen.evaluate(3.0, 1), 3.0)
    
    def test_p2_polynomial(self):
        """Test P_2(x) = (3x^2 - 1)/2"""
        coeffs = self.gen.get_coefficients(2)
        # Coeffs in descending order: [1.5, 0.0, -0.5] for 1.5*x^2 + 0*x - 0.5
        expected = [1.5, 0.0, -0.5]
        self.assertAlmostEqual(coeffs[0], expected[0])
        self.assertAlmostEqual(coeffs[1], expected[1])
        self.assertAlmostEqual(coeffs[2], expected[2])
    
    def test_p3_polynomial(self):
        """Test P_3(x) = (5x^3 - 3x)/2"""
        coeffs = self.gen.get_coefficients(3)
        # Coeffs in descending order: [2.5, 0.0, -1.5, 0.0]
        expected = [2.5, 0.0, -1.5, 0.0]
        for i, (c, e) in enumerate(zip(coeffs, expected)):
            self.assertAlmostEqual(c, e)
    
    def test_p4_polynomial(self):
        """Test P_4(x) = (35x^4 - 30x^2 + 3)/8"""
        coeffs = self.gen.get_coefficients(4)
        expected = [35/8, 0.0, -30/8, 0.0, 3/8]
        for i, (c, e) in enumerate(zip(coeffs, expected)):
            self.assertAlmostEqual(c, e)
    
    def test_orthogonality(self):
        """Test orthogonality property: integral of P_n*P_m over [-1,1] = 0 for n != m."""
        from scipy.integrate import quad
        
        # Test a few pairs
        pairs = [(0, 1), (1, 2), (2, 3)]
        for n, m in pairs:
            integrand = lambda x: self.gen.evaluate(x, n) * self.gen.evaluate(x, m)
            result, _ = quad(integrand, -1, 1)
            self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_normalization(self):
        """Test normalization: integral of P_n^2 over [-1,1] = 2/(2n+1)."""
        from scipy.integrate import quad
        
        for n in range(5):
            integrand = lambda x: self.gen.evaluate(x, n) ** 2
            result, _ = quad(integrand, -1, 1)
            expected = 2.0 / (2*n + 1)
            self.assertAlmostEqual(result, expected, places=5)
    
    def test_boundary_values(self):
        """Test P_n(1) = 1 and P_n(-1) = (-1)^n."""
        for n in range(10):
            self.assertAlmostEqual(self.gen.evaluate(1.0, n), 1.0)
            expected = (-1) ** n
            self.assertAlmostEqual(self.gen.evaluate(-1.0, n), expected)
    
    def test_derivative_p2(self):
        """Test derivative of P_2(x) = (3x^2 - 1)/2 is d/dx = 3x."""
        deriv_coeffs = self.gen.derivative_coefficients(2)
        # Derivative should be [3.0, 0.0] for 3*x in descending order
        self.assertAlmostEqual(deriv_coeffs[0], 3.0)
        self.assertAlmostEqual(deriv_coeffs[1], 0.0)
    
    def test_integral_p1(self):
        """Test integral of P_1(x) = x is ?x dx = x^2/2."""
        int_coeffs = self.gen.integral_coefficients(1)
        # Integral should be [0.5, 0.0, 0.0] for 0.5*x^2 + C in descending order
        self.assertAlmostEqual(int_coeffs[0], 0.5)
        self.assertAlmostEqual(int_coeffs[1], 0.0)
        self.assertAlmostEqual(int_coeffs[2], 0.0)
    
    def test_generate_basis(self):
        """Test generating a basis of Legendre polynomials."""
        max_n = 5
        basis = self.gen.generate_basis(max_n)
        self.assertEqual(len(basis), max_n + 1)
        for i, coeffs in enumerate(basis):
            self.assertEqual(len(coeffs), i + 1)


class TestLegendreSympy(unittest.TestCase):
    """Test suite for Sympy integration."""
    
    def test_sympy_available(self):
        """Check if SymPy is available."""
        from legendre.sympy_integration import SYMPY_AVAILABLE
        self.assertTrue(SYMPY_AVAILABLE, "SymPy should be installed for these tests")
    
    def test_generate_sympy_legendre_p0(self):
        """Test generate_sympy_legendre for n=0."""
        from legendre.sympy_integration import generate_sympy_legendre
        poly = generate_sympy_legendre(0)
        self.assertEqual(str(poly), '1')
    
    def test_generate_sympy_legendre_p1(self):
        """Test generate_sympy_legendre for n=1."""
        from legendre.sympy_integration import generate_sympy_legendre
        poly = generate_sympy_legendre(1)
        self.assertEqual(str(poly), 'x')
    
    def test_generate_sympy_legendre_p2(self):
        """Test generate_sympy_legendre for n=2."""
        from legendre.sympy_integration import generate_sympy_legendre
        poly = generate_sympy_legendre(2)
        # P_2(x) = (3x^2 - 1)/2
        self.assertIn('3*x**2', str(poly))
    
    def test_get_derivative(self):
        """Test derivative computation."""
        from legendre.sympy_integration import SympyLegendreGenerator
        gen = SympyLegendreGenerator()
        deriv = gen.get_derivative(2)
        # d/dx((3x^2 - 1)/2) = 3x
        self.assertEqual(str(deriv), '3*x')
    
    def test_get_integral(self):
        """Test integral computation."""
        from legendre.sympy_integration import SympyLegendreGenerator
        gen = SympyLegendreGenerator()
        integ = gen.get_integral(1)
        # ?x dx = x^2/2
        self.assertEqual(str(integ), 'x**2/2')
    
    def test_evaluate(self):
        """Test evaluation at specific points."""
        from legendre.sympy_integration import SympyLegendreGenerator
        gen = SympyLegendreGenerator()
        result = gen.evaluate(0.5, 2)
        # P_2(0.5) = (3*(0.5)^2 - 1)/2 = (0.75 - 1)/2 = -0.125
        self.assertAlmostEqual(result, -0.125)
    
    def test_get_sympy_legendre_basis(self):
        """Test get_sympy_legendre_basis function."""
        from legendre.sympy_integration import get_sympy_legendre_basis
        basis = get_sympy_legendre_basis(3)
        self.assertEqual(len(basis), 4)


class TestLegendreNumpy(unittest.TestCase):
    """Test suite for NumPy integration."""
    
    def test_numpy_available(self):
        """Check if NumPy is available."""
        from legendre.numpy_integration import NumpyLegendreGenerator
        gen = NumpyLegendreGenerator()
        self.assertIsNotNone(gen)
    
    def test_get_coefficients_p0(self):
        """Test get_coefficients for n=0."""
        from legendre.numpy_integration import generate_numpy_legendre
        coeffs = generate_numpy_legendre(0)
        self.assertEqual(len(coeffs), 1)
        self.assertAlmostEqual(float(coeffs[0]), 1.0)
    
    def test_get_coefficients_p1(self):
        """Test get_coefficients for n=1."""
        from legendre.numpy_integration import generate_numpy_legendre_descending
        coeffs = generate_numpy_legendre_descending(1)
        self.assertEqual(len(coeffs), 2)
        self.assertAlmostEqual(float(coeffs[0]), 1.0)  # x term (descending order)
        self.assertAlmostEqual(float(coeffs[1]), 0.0)  # constant term
    
    def test_evaluate(self):
        """Test evaluation at specific points."""
        from legendre.numpy_integration import evaluate_numpy_legendre
        result = evaluate_numpy_legendre(0.5, 2)
        # P_2(0.5) = (3*(0.5)^2 - 1)/2 = -0.125
        self.assertAlmostEqual(result, -0.125)
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        from legendre.numpy_integration import NumpyLegendreGenerator
        gen = NumpyLegendreGenerator()
        xs = [0.0, 0.5, 1.0]
        results = gen.evaluate_batch(xs, 2)
        self.assertEqual(len(results), 3)
    
    def test_generate_basis(self):
        """Test generating a basis."""
        from legendre.numpy_integration import get_numpy_legendre_basis
        basis = get_numpy_legendre_basis(4)
        self.assertEqual(len(basis), 5)
        for i, coeffs in enumerate(basis):
            self.assertEqual(len(coeffs), i + 1)


class TestLegendreMpmath(unittest.TestCase):
    """Test suite for mpmath integration."""
    
    def test_mpmath_available(self):
        """Check if mpmath is available."""
        from legendre.mpmath_integration import MPMATH_AVAILABLE
        self.assertTrue(MPMATH_AVAILABLE, "mpmath should be installed for these tests")
    
    def test_evaluate_p0(self):
        """Test evaluation of P_0(x)."""
        from legendre.mpmath_integration import evaluate_mpmath_legendre
        result = evaluate_mpmath_legendre('0.5', 0)
        # Result is mp.mpf('1'), use float comparison for robustness
        self.assertAlmostEqual(float(result), 1.0)
    
    def test_evaluate_p1(self):
        """Test evaluation of P_1(x)."""
        from legendre.mpmath_integration import evaluate_mpmath_legendre
        result = evaluate_mpmath_legendre('0.5', 1)
        self.assertEqual(str(result), '0.5')
    
    def test_evaluate_p2(self):
        """Test evaluation of P_2(x)."""
        from legendre.mpmath_integration import evaluate_mpmath_legendre
        result = evaluate_mpmath_legendre('0.5', 2)
        # P_2(0.5) = (3*(0.5)^2 - 1)/2 = -0.125
        self.assertAlmostEqual(float(result), -0.125)
    
    def test_high_precision(self):
        """Test high precision evaluation."""
        from legendre.mpmath_integration import evaluate_mpmath_legendre
        result = evaluate_mpmath_legendre('0.333333333333333', 1, dps=50)
        self.assertEqual(str(result), '0.333333333333333')
    
    def test_generate_basis(self):
        """Test generating a basis."""
        from legendre.mpmath_integration import get_mpmath_legendre_basis
        basis = get_mpmath_legendre_basis(3)
        self.assertEqual(len(basis), 4)
        # Each element should be callable
        for poly in basis:
            result = poly('0.5')
            self.assertIsNotNone(result)


class TestIntegration(unittest.TestCase):
    """Integration tests comparing different implementations."""
    
    def test_core_vs_numpy(self):
        """Compare core generator with numpy implementation."""
        from legendre.core import LegendreGenerator
        from legendre.numpy_integration import evaluate_numpy_legendre
        
        gen = LegendreGenerator()
        for n in range(5):
            for x in [0.0, 0.5, -0.5, 1.0]:
                core_result = gen.evaluate(x, n)
                numpy_result = evaluate_numpy_legendre(x, n)
                self.assertAlmostEqual(core_result, numpy_result, places=10)
    
    def test_core_vs_sympy(self):
        """Compare core generator with sympy implementation."""
        from legendre.core import LegendreGenerator
        from legendre.sympy_integration import generate_sympy_legendre
        
        gen = LegendreGenerator()
        for n in range(5):
            for x in [0.0, 0.5, -0.5]:
                core_result = gen.evaluate(x, n)
                sympy_poly = generate_sympy_legendre(n)
                sympy_result = float(sympy_poly.subs('x', x))
                self.assertAlmostEqual(core_result, sympy_result, places=10)
    
    def test_core_vs_mpmath(self):
        """Compare core generator with mpmath implementation."""
        from legendre.core import LegendreGenerator
        from legendre.mpmath_integration import evaluate_mpmath_legendre
        
        gen = LegendreGenerator()
        for n in range(5):
            for x in [0.0, 0.5]:
                core_result = gen.evaluate(x, n)
                mpmath_result = float(evaluate_mpmath_legendre(str(x), n))
                self.assertAlmostEqual(core_result, mpmath_result, places=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)

