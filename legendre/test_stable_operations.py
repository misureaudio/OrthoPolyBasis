"""
Tests for Stable Legendre Operations and Clenshaw Evaluation
=============================================================
"""

import unittest
import numpy as np
from scipy.special import eval_legendre


class TestLegendreBasisDerivative(unittest.TestCase):
    """Test derivative operations in Legendre basis."""
    
    def test_derivative_p0(self):
        """d/dx P_0 = d/dx(1) = 0"""
        from legendre.stable_operations import derivative_legendre
        coeffs = np.array([1.0])  # P_0
        result = derivative_legendre(coeffs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result[0]), 0.0)
    
    def test_derivative_p1(self):
        """d/dx P_1 = d/dx(x) = 1 = P_0"""
        from legendre.stable_operations import derivative_legendre
        coeffs = np.array([0.0, 1.0])  # P_1 (coefficient of P_0 is 0, P_1 is 1)
        result = derivative_legendre(coeffs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result[0]), 1.0)  # Result is P_0
    
    def test_derivative_p2(self):
        """d/dx P_2 = d/dx((3x²-1)/2) = 3x = 3*P_1"""
        from legendre.stable_operations import derivative_legendre
        coeffs = np.array([0.0, 0.0, 1.0])  # P_2
        result = derivative_legendre(coeffs)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(float(result[0]), 0.0)   # No P_0 term
        self.assertAlmostEqual(float(result[1]), 3.0)   # 3*P_1
    
    def test_derivative_p3(self):
        """d/dx P_3 = d/dx((5x³-3x)/2) = (15x²-3)/2 = (5/2)*(3x²-1) + 0 = 5*P_2 + P_0
        Actually: (15x²-3)/2 = 15/2 * x² - 3/2 = 15/6*(3x²-1) + 15/6 - 3/2 = 5/2*P_2 + 5/2 - 3/2
        = 5/2*P_2 + 1 = P_0 + (5/2)*P_2... let me recalculate.
        
        P_3 = (5x³-3x)/2
        d/dx P_3 = (15x²-3)/2
        
        We want: (15x²-3)/2 = a*P_0 + b*P_1 + c*P_2
                 = a + b*x + c*(3x²-1)/2
                 = a - c/2 + b*x + (3c/2)*x²
        
        Matching: 3c/2 = 15/2 => c = 5
                   b = 0
                   a - c/2 = -3/2 => a - 5/2 = -3/2 => a = 1
        
        So d/dx P_3 = P_0 + 5*P_2
        """
        from legendre.stable_operations import derivative_legendre
        coeffs = np.array([0.0, 0.0, 0.0, 1.0])  # P_3
        result = derivative_legendre(coeffs)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(float(result[0]), 1.0)   # P_0 coefficient
        self.assertAlmostEqual(float(result[1]), 0.0)   # No P_1 term
        self.assertAlmostEqual(float(result[2]), 5.0)   # 5*P_2
    
    def test_derivative_numerical_verification(self):
        """Verify derivative numerically using finite differences."""
        from legendre.stable_operations import LegendreBasisOperations
        from legendre.clenshaw_evaluation import evaluate_legendre_series
        
        # Test polynomial: 2*P_0 + 3*P_1 - P_2 + 0.5*P_3
        coeffs = np.array([2.0, 3.0, -1.0, 0.5])
        deriv_coeffs = LegendreBasisOperations.derivative_legendre_basis(coeffs)
        
        h = 1e-6
        for x in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            # Numerical derivative via finite difference
            f_plus = evaluate_legendre_series(coeffs, x + h)
            f_minus = evaluate_legendre_series(coeffs, x - h)
            numerical_deriv = (f_plus - f_minus) / (2*h)
            
            # Analytical derivative
            analytical_deriv = evaluate_legendre_series(deriv_coeffs, x)
            
            self.assertAlmostEqual(numerical_deriv, analytical_deriv, places=5,
                msg=f"Mismatch at x={x}")


class TestLegendreBasisIntegral(unittest.TestCase):
    """Test integral operations in Legendre basis."""
    
    def test_integral_p0(self):
        """int P_0 dx = int 1 dx = x = P_1"""
        from legendre.stable_operations import integral_legendre
        coeffs = np.array([1.0])  # P_0
        result = integral_legendre(coeffs)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(float(result[0]), 0.0)   # No constant term (C=0)
        self.assertAlmostEqual(float(result[1]), 1.0)   # P_1 coefficient
    
    def test_integral_p1(self):
        """int P_1 dx = int x dx = x²/2
        Using formula: int(P_1) = (P_2 - P_0)/3
        So result should be [-1/3, 0, 1/3]
        """
        from legendre.stable_operations import integral_legendre
        coeffs = np.array([0.0, 1.0])  # P_1
        result = integral_legendre(coeffs)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(float(result[0]), -1.0/3.0)   # -P_0/3
        self.assertAlmostEqual(float(result[1]), 0.0)       # No P_1 term
        self.assertAlmostEqual(float(result[2]), 1.0/3.0)   # P_2/3
    
    def test_integral_with_constant(self):
        """Test constant of integration."""
        from legendre.stable_operations import integral_legendre
        coeffs = np.array([1.0])  # P_0
        result = integral_legendre(coeffs, C=5.0)
        self.assertAlmostEqual(float(result[0]), 5.0)   # Constant term
    
    def test_integral_numerical_verification(self):
        """Verify integral numerically."""
        from legendre.stable_operations import LegendreBasisOperations
        from legendre.clenshaw_evaluation import evaluate_legendre_series
        from scipy.integrate import quad
        
        # Test polynomial: P_1 + 2*P_2
        coeffs = np.array([0.0, 1.0, 2.0])
        int_coeffs = LegendreBasisOperations.integral_legendre_basis(coeffs, C=0)
        
        for x in [-0.8, -0.3, 0.3, 0.8]:
            # Numerical integral from 0 to x
            numerical_int, _ = quad(lambda t: evaluate_legendre_series(coeffs, t), 0, x)
            
            # Analytical integral (value at x minus value at 0)
            analytical_at_x = evaluate_legendre_series(int_coeffs, x)
            analytical_at_0 = evaluate_legendre_series(int_coeffs, 0)
            analytical_int = analytical_at_x - analytical_at_0
            
            self.assertAlmostEqual(numerical_int, analytical_int, places=5,
                msg=f"Mismatch at x={x}")


class TestClenshawEvaluation(unittest.TestCase):
    """Test Clenshaw evaluation algorithm."""
    
    def test_clenshaw_single_term(self):
        """Evaluate single Legendre polynomial."""
        from legendre.clenshaw_evaluation import evaluate_legendre_series
        
        for n in range(10):
            coeffs = np.zeros(n + 1)
            coeffs[n] = 1.0  # Just P_n
            
            for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                result = evaluate_legendre_series(coeffs, x)
                expected = eval_legendre(n, x)
                self.assertAlmostEqual(result, expected, places=12,
                    msg=f"P_{n}({x}) mismatch")
    
    def test_clenshaw_linear_combination(self):
        """Evaluate linear combination of Legendre polynomials."""
        from legendre.clenshaw_evaluation import evaluate_legendre_series
        
        # Test: 2*P_0 - 3*P_1 + P_2 - 0.5*P_3
        coeffs = np.array([2.0, -3.0, 1.0, -0.5])
        
        for x in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
            result = evaluate_legendre_series(coeffs, x)
            
            # Direct computation for verification
            expected = sum(coeffs[k] * eval_legendre(k, x) for k in range(len(coeffs)))
            
            self.assertAlmostEqual(result, expected, places=12,
                msg=f"Mismatch at x={x}")
    
    def test_clenshaw_vs_naive_evaluation(self):
        """Compare Clenshaw with naive summation (should give same result)."""
        from legendre.clenshaw_evaluation import evaluate_legendre_series
        
        np.random.seed(42)
        for n in [5, 10, 20]:
            coeffs = np.random.randn(n + 1)
            
            for x in [-0.9, -0.3, 0.3, 0.9]:
                clenshaw_result = evaluate_legendre_series(coeffs, x)
                naive_result = sum(coeffs[k] * eval_legendre(k, x) for k in range(n + 1))
                
                self.assertAlmostEqual(clenshaw_result, naive_result, places=12,
                    msg=f"n={n}, x={x} mismatch")
    
    def test_clenshaw_high_degree(self):
        """Test evaluation of high-degree series (where stability matters)."""
        from legendre.clenshaw_evaluation import evaluate_legendre_series
        
        # High degree with alternating coefficients (stress test)
        n = 50
        coeffs = np.array([((-1)**k) / (k + 1) for k in range(n + 1)])
        
        for x in [-0.99, 0.0, 0.99]:
            result = evaluate_legendre_series(coeffs, x)
            expected = sum(coeffs[k] * eval_legendre(k, x) for k in range(n + 1))
            
            # Allow slightly less precision for high degree
            self.assertAlmostEqual(result, expected, places=10,
                msg=f"High-degree mismatch at x={x}")
    
    def test_clenshaw_batch_evaluation(self):
        """Test evaluation at multiple points."""
        from legendre.clenshaw_evaluation import evaluate_legendre_series
        
        coeffs = np.array([1.0, 2.0, 3.0, 4.0])
        xs = np.linspace(-1, 1, 11)
        
        results = evaluate_legendre_series(coeffs, xs)
        
        self.assertEqual(len(results), len(xs))
        for x, r in zip(xs, results):
            expected = sum(coeffs[k] * eval_legendre(k, x) for k in range(len(coeffs)))
            self.assertAlmostEqual(r, expected, places=12)


class TestBasisConversion(unittest.TestCase):
    """Test conversion between monomial and Legendre bases."""
    
    def test_monomial_to_legendre_identity(self):
        """Convert 1 = P_0."""
        from legendre.stable_operations import LegendreBasisOperations
        mono = np.array([1.0])  # Constant 1
        leg = LegendreBasisOperations.convert_from_monomial(mono)
        self.assertAlmostEqual(float(leg[0]), 1.0)
    
    def test_monomial_to_legendre_x(self):
        """Convert x = P_1."""
        from legendre.stable_operations import LegendreBasisOperations
        mono = np.array([0.0, 1.0])  # x
        leg = LegendreBasisOperations.convert_from_monomial(mono)
        self.assertAlmostEqual(float(leg[0]), 0.0)
        self.assertAlmostEqual(float(leg[1]), 1.0)
    
    def test_monomial_to_legendre_x2(self):
        """Convert x² = (2*P_2 + P_0)/3."""
        from legendre.stable_operations import LegendreBasisOperations
        mono = np.array([0.0, 0.0, 1.0])  # x²
        leg = LegendreBasisOperations.convert_from_monomial(mono)
        self.assertAlmostEqual(float(leg[0]), 1.0/3.0)   # P_0 coefficient
        self.assertAlmostEqual(float(leg[1]), 0.0)       # No P_1 term
        self.assertAlmostEqual(float(leg[2]), 2.0/3.0)   # P_2 coefficient
    
    def test_roundtrip_conversion(self):
        """Convert monomial -> Legendre -> monomial should recover original."""
        from legendre.stable_operations import LegendreBasisOperations
        
        mono_original = np.array([1.0, 2.0, 3.0, 4.0])  # 1 + 2x + 3x² + 4x³
        leg = LegendreBasisOperations.convert_from_monomial(mono_original)
        mono_recovered = LegendreBasisOperations.convert_to_monomial(leg)
        
        for i, (orig, rec) in enumerate(zip(mono_original, mono_recovered)):
            self.assertAlmostEqual(float(orig), float(rec), places=10,
                msg=f"Coefficient {i} mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
