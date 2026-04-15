import unittest
import math
import numpy as np
from sympy import Rational, factorial, expand

# ============================================================================
# Import handling for multiple deployment scenarios
# ============================================================================
# This test file supports two modes:
# 1. Package mode: laguerre is installed as a package (import laguerre)
# 2. Development mode: running directly from source directory
#
# Since the laguerre modules use relative imports internally (from .polynomial),
# they MUST be imported as a package, not as standalone modules.
# ============================================================================

try:
    # Mode 1: Package is installed - import normally
    from laguerre.polynomial import LaguerrePolynomial, GeneralizedLaguerrePolynomial
    from laguerre.basis import (
        LaguerreBasis,
        GeneralizedLaguerreBasis,
        compute_roots,
        gauss_quadrature_weights,
        function_projection,
        function_approximation
    )
    from laguerre.utils import (
        to_sympy, evaluate_array, generate_basis_matrix,
        evaluate_mp, stable_evaluation, condition_estimate
    )
except ImportError:
    # Mode 2: Development mode - set up path so laguerre can be imported as package
    import sys
    from pathlib import Path
    
    # Get the directory containing this test file (the laguerre package directory)
    test_dir = Path(__file__).resolve().parent
    parent_dir = test_dir.parent  # This should contain the 'laguerre' directory
    
    # Add parent directory to sys.path so Python can find 'laguerre' as a package
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Now import as a package (this works because modules use relative imports)
    from laguerre.polynomial import LaguerrePolynomial, GeneralizedLaguerrePolynomial
    from laguerre.basis import (
        LaguerreBasis,
        GeneralizedLaguerreBasis,
        compute_roots,
        gauss_quadrature_weights,
        function_projection,
        function_approximation
    )
    from laguerre.utils import (
        to_sympy, evaluate_array, generate_basis_matrix,
        evaluate_mp, stable_evaluation, condition_estimate
    )


class TestLaguerrePolynomial(unittest.TestCase):
    """Tests for Laguerre polynomial class."""

    def test_constructor(self):
        """Test basic construction."""
        L5 = LaguerrePolynomial(5)
        self.assertEqual(L5.n, 5)
        self.assertEqual(L5.degree, 5)

    def test_degree_zero(self):
        """Test L_0(x) = 1."""
        L0 = LaguerrePolynomial(0)
        coeffs = L0.get_coefficients()
        self.assertEqual(len(coeffs), 1)
        self.assertAlmostEqual(coeffs[0], 1.0, places=14)

    def test_degree_one(self):
        """Test L_1(x) = 1 - x."""
        L1 = LaguerrePolynomial(1)
        coeffs = L1.get_coefficients()
        self.assertEqual(len(coeffs), 2)
        self.assertAlmostEqual(coeffs[0], 1.0, places=14)   # constant
        self.assertAlmostEqual(coeffs[1], -1.0, places=14)  # x coefficient

    def test_degree_two(self):
        """Test L_2(x) = (1/2)(x^2 - 4x + 2)."""
        L2 = LaguerrePolynomial(2)
        coeffs = L2.get_coefficients()
        self.assertEqual(len(coeffs), 3)
        # Coefficients: [constant, x, x^2] in ascending order
        self.assertAlmostEqual(coeffs[0], 1.0, places=14)    # constant term (after scaling)
        self.assertAlmostEqual(coeffs[1], -2.0, places=14)   # x coefficient
        self.assertAlmostEqual(coeffs[2], 0.5, places=14)    # x^2 coefficient

    def test_degree_three(self):
        """Test L_3(x) = (1/6)(-x^3 + 9x^2 - 18x + 6)."""
        L3 = LaguerrePolynomial(3)
        coeffs = L3.get_coefficients()
        self.assertEqual(len(coeffs), 4)
        # Coefficients in ascending order
        self.assertAlmostEqual(coeffs[0], 1.0, places=14)     # constant
        self.assertAlmostEqual(coeffs[1], -3.0, places=14)    # x
        self.assertAlmostEqual(coeffs[2], 1.5, places=14)     # x^2
        self.assertAlmostEqual(coeffs[3], -1.0/6.0, places=14) # x^3

    def test_evaluate_at_zero(self):
        """Test L_n(0) = 1 for all n."""
        for n in range(20):
            Ln = LaguerrePolynomial(n)
            value = Ln.evaluate(0)
            self.assertAlmostEqual(value, 1.0, places=10,
                                   msg=f"L_{n}(0) should equal 1")

    def test_evaluate_at_one(self):
        """Test L_n(1) values."""
        # Known values: L_0(1)=1, L_1(1)=0, L_2(1)=-0.5, L_3(1)=-1/6
        expected = [1.0, 0.0, -0.5, -1.0/6.0]
        for n in range(len(expected)):
            Ln = LaguerrePolynomial(n)
            value = Ln.evaluate(1)
            self.assertAlmostEqual(value, expected[n], places=10,
                msg=f"L_{n}(1) should equal {expected[n]}")

    def test_call_interface(self):
        """Test __call__ interface."""
        L5 = LaguerrePolynomial(5)
        self.assertEqual(L5.evaluate(2.0), L5(2.0))

    def test_negative_degree_raises(self):
        """Test that negative degree raises ValueError."""
        with self.assertRaises(ValueError):
            LaguerrePolynomial(-1)

    def test_coefficients_caching(self):
        """Test that coefficients are cached after first computation."""
        L5 = LaguerrePolynomial(5)
        coeffs1 = L5.get_coefficients()
        coeffs2 = L5.get_coefficients()
        # Should return copies, not same object
        self.assertIsNot(coeffs1, coeffs2)
        self.assertEqual(coeffs1, coeffs2)


class TestGeneralizedLaguerrePolynomial(unittest.TestCase):
    """Tests for generalized Laguerre polynomial class."""

    def test_constructor_with_alpha(self):
        """Test construction with alpha parameter."""
        L5 = GeneralizedLaguerrePolynomial(5, alpha=2.0)
        self.assertEqual(L5.n, 5)
        self.assertEqual(L5.alpha, 2.0)

    def test_alpha_zero_equals_standard(self):
        """Test that alpha=0 gives standard Laguerre polynomials."""
        for n in range(10):
            L_std = LaguerrePolynomial(n)
            L_gen = GeneralizedLaguerrePolynomial(n, alpha=0.0)

            # Compare coefficients
            coeffs_std = L_std.get_coefficients()
            coeffs_gen = L_gen.get_coefficients()
            self.assertEqual(len(coeffs_std), len(coeffs_gen))
            for c1, c2 in zip(coeffs_std, coeffs_gen):
                self.assertAlmostEqual(c1, c2, places=14)

    def test_degree_zero_alpha(self):
        """Test L_0^(alpha)(x) = 1."""
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            L0 = GeneralizedLaguerrePolynomial(0, alpha=alpha)
            coeffs = L0.get_coefficients()
            self.assertEqual(len(coeffs), 1)
            self.assertAlmostEqual(coeffs[0], 1.0, places=14)

    def test_degree_one_alpha(self):
        """Test L_1^(alpha)(x) = alpha + 1 - x."""
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            L1 = GeneralizedLaguerrePolynomial(1, alpha=alpha)
            coeffs = L1.get_coefficients()
            self.assertEqual(len(coeffs), 2)
            self.assertAlmostEqual(coeffs[0], alpha + 1, places=14)  # constant
            self.assertAlmostEqual(coeffs[1], -1.0, places=14)       # x coefficient

    def test_evaluate_at_zero_alpha(self):
        """Test L_n^(alpha)(0) = binomial(n+alpha, n)."""
        for alpha in [0.0, 0.5, 1.0]:
            for n in range(5):
                Ln = GeneralizedLaguerrePolynomial(n, alpha=alpha)
                value = Ln.evaluate(0)
                # L_n^(alpha)(0) = (n+alpha)!/(n!*alpha!) = binomial(n+alpha, n)
                expected = math.gamma(n + alpha + 1) / (math.gamma(n + 1) * math.gamma(alpha + 1))
                self.assertAlmostEqual(value, expected, places=10,
                    msg=f"L_{n}^{({alpha})}(0) should equal {expected}")

    def test_alpha_negative_one_raises(self):
        """Test that alpha <= -1 raises ValueError."""
        with self.assertRaises(ValueError):
            GeneralizedLaguerrePolynomial(5, alpha=-1)
        with self.assertRaises(ValueError):
            GeneralizedLaguerrePolynomial(5, alpha=-2)


class TestRecurrenceRelations(unittest.TestCase):
    """Tests for recurrence relations."""

    def test_standard_recurrence(self):
        """Test (n+1)L_{n+1} = (2n+1-x)L_n - nL_{n-1}."""
        x_test = np.linspace(0, 10, 50)

        for n in range(1, 20):
            Ln_minus_1 = LaguerrePolynomial(n-1)
            Ln = LaguerrePolynomial(n)
            Ln_plus_1 = LaguerrePolynomial(n+1)

            # Compute recurrence
            recurrence = ((2*n + 1 - x_test) * Ln.evaluate(x_test) - n * Ln_minus_1.evaluate(x_test)) / (n + 1)
            actual = Ln_plus_1.evaluate(x_test)

            np.testing.assert_array_almost_equal(recurrence, actual, decimal=10)

    def test_generalized_recurrence(self):
        """Test (n+1)L_{n+1}^(alpha) = (2n+alpha+1-x)L_n^(alpha) - (n+alpha)L_{n-1}^(alpha)."""
        x_test = np.linspace(0, 10, 50)

        for alpha in [0.0, 0.5, 1.0, 2.0]:
            for n in range(1, 15):
                Ln_minus_1 = GeneralizedLaguerrePolynomial(n-1, alpha=alpha)
                Ln = GeneralizedLaguerrePolynomial(n, alpha=alpha)
                Ln_plus_1 = GeneralizedLaguerrePolynomial(n+1, alpha=alpha)

                # Compute recurrence
                numerator = (2*n + alpha + 1 - x_test) * Ln.evaluate(x_test) - (n + alpha) * Ln_minus_1.evaluate(x_test)
                recurrence = numerator / (n + 1)
                actual = Ln_plus_1.evaluate(x_test)

                np.testing.assert_array_almost_equal(recurrence, actual, decimal=10)


class TestOrthogonality(unittest.TestCase):
    """Tests for orthogonality properties."""

    def test_norm_squared_standard(self):
        """Test ||L_n||^2 = 1 for standard Laguerre polynomials."""
        basis = LaguerreBasis(20)

        for n in range(20):
            norm_sq = basis.norm_squared(n)
            self.assertAlmostEqual(norm_sq, 1.0, places=10,
                msg=f"||L_{n}||^2 should equal 1")

    def test_norm_squared_generalized(self):
        """Test ||L_n^(alpha)||^2 = gamma(n+alpha+1)/gamma(n+1)."""
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            basis = GeneralizedLaguerreBasis(15, alpha=alpha)

            for n in range(15):
                norm_sq = basis.norm_squared(n)
                expected = math.gamma(n + alpha + 1) / math.gamma(n + 1)
                self.assertAlmostEqual(norm_sq, expected, places=8,
                    msg=f"||L_{n}^{({alpha})}||^2 should equal {expected}")

    def test_orthogonality_standard(self):
        """Test orthogonality of standard Laguerre polynomials."""
        basis = LaguerreBasis(15)

        for m in range(min(12, basis.max_degree + 1)):
            for n in range(m + 1, min(12, basis.max_degree + 1)):
                inner_product = basis.inner_product(
                    lambda x, p=basis[m]: p(x),
                    lambda x, p=basis[n]: p(x)
                )
                # Use degree-dependent tolerance for numerical stability
                max_deg = max(m, n)
                tol = 1e-6 * (2 ** (max_deg // 3))
                self.assertLess(abs(inner_product), tol,
                    f"L_{m} and L_{n} should be orthogonal")

    def test_orthogonality_generalized(self):
        """Test orthogonality of generalized Laguerre polynomials."""
        for alpha in [0.0, 0.5, 1.0]:
            basis = GeneralizedLaguerreBasis(12, alpha=alpha)

            for m in range(min(10, basis.max_degree + 1)):
                for n in range(m + 1, min(10, basis.max_degree + 1)):
                    inner_product = basis.inner_product(
                        lambda x, p=basis[m]: p(x),
                        lambda x, p=basis[n]: p(x)
                    )
                    max_deg = max(m, n)
                    tol = 1e-5 * (2 ** (max_deg // 3))
                    self.assertLess(abs(inner_product), tol,
                        f"L_{m}^{({alpha})} and L_{n}^{({alpha})} should be orthogonal")


class TestQuadrature(unittest.TestCase):
    """Tests for Gauss-Laguerre quadrature."""

    def test_compute_roots(self):
        """Test root computation."""
        roots = compute_roots(5)
        self.assertEqual(len(roots), 5)
        # All roots should be positive (domain is [0, +inf))
        for r in roots:
            self.assertGreater(r, 0)

    def test_compute_roots_generalized(self):
        """Test root computation with alpha."""
        roots = compute_roots(5, alpha=1.0)
        self.assertEqual(len(roots), 5)
        for r in roots:
            self.assertGreater(r, 0)

    def test_gauss_quadrature_weights(self):
        """Test weight computation."""
        pts, wts = gauss_quadrature_weights(5)
        self.assertEqual(len(pts), 5)
        self.assertEqual(len(wts), 5)
        # All weights should be positive
        for w in wts:
            self.assertGreater(w, 0)

    def test_integrate_constant(self):
        """Test integration of constant function."""
        basis = LaguerreBasis(20)
        pts, wts = basis._gauss_quadrature(20)

        # integral of 1 * exp(-x) dx from 0 to inf = 1
        result = sum(w for w in wts)
        self.assertAlmostEqual(result, 1.0, places=8)

    def test_integrate_polynomial(self):
        """Test integration of polynomial."""
        basis = LaguerreBasis(20)
        pts, wts = basis._gauss_quadrature(20)

        # integral of x * exp(-x) dx from 0 to inf = 1
        result = sum(x * w for x, w in zip(pts, wts))
        self.assertAlmostEqual(result, 1.0, places=8)

    def test_integrate_x_squared(self):
        """Test integration of x^2."""
        basis = LaguerreBasis(20)
        pts, wts = basis._gauss_quadrature(20)

        # integral of x^2 * exp(-x) dx from 0 to inf = 2
        result = sum(x**2 * w for x, w in zip(pts, wts))
        self.assertAlmostEqual(result, 2.0, places=8)

    def test_integrate_x_to_n(self):
        """Test integration of x^n."""
        basis = LaguerreBasis(30)
        pts, wts = basis._gauss_quadrature(30)

        # integral of x^n * exp(-x) dx from 0 to inf = n!
        for n in range(10):
            result = sum(x**n * w for x, w in zip(pts, wts))
            expected = math.factorial(n)
            self.assertAlmostEqual(result, expected, places=6,
                msg=f"integral of x^{n} should equal {expected}")


class TestBasisOperations(unittest.TestCase):
    """Tests for basis class operations."""

    def test_laguerre_basis_construction(self):
        """Test LaguerreBasis construction."""
        basis = LaguerreBasis(10)
        self.assertEqual(basis.max_degree, 10)
        self.assertEqual(len(basis), 11)  # degrees 0 to 10

    def test_generalized_basis_construction(self):
        """Test GeneralizedLaguerreBasis construction."""
        basis = GeneralizedLaguerreBasis(10, alpha=1.5)
        self.assertEqual(basis.max_degree, 10)
        self.assertEqual(len(basis), 11)

    def test_basis_indexing(self):
        """Test basis indexing."""
        basis = LaguerreBasis(10)
        for n in range(11):
            P = basis[n]
            self.assertEqual(P.n, n)

    def test_basis_iteration(self):
        """Test basis iteration."""
        basis = LaguerreBasis(5)
        degrees = [P.n for P in basis]
        self.assertEqual(degrees, list(range(6)))

    def test_weight_function_standard(self):
        """Test weight function e^(-x)."""
        basis = LaguerreBasis(10)
        for x in [0.0, 1.0, 2.0, 5.0]:
            w = basis.weight_function(x)
            expected = math.exp(-x)
            self.assertAlmostEqual(w, expected, places=14)

    def test_weight_function_generalized(self):
        """Test weight function x^alpha * e^(-x)."""
        alpha = 1.5
        basis = GeneralizedLaguerreBasis(10, alpha=alpha)
        for x in [0.5, 1.0, 2.0, 5.0]:
            w = basis.weight_function(x)
            expected = (x**alpha) * math.exp(-x)
            self.assertAlmostEqual(w, expected, places=14)

    def test_projection_constant(self):
        """Test projection of constant function."""
        basis = LaguerreBasis(10)
        f = lambda x: 1.0
        coeffs = basis.project(f)

        # L_0 = 1, so projecting 1 should give [1, 0, 0, ...]
        self.assertAlmostEqual(coeffs[0], 1.0, places=8)
        for n in range(1, len(coeffs)):
            self.assertLess(abs(coeffs[n]), 1e-6)

    def test_projection_linear(self):
        """Test projection of linear function."""
        basis = LaguerreBasis(10)
        f = lambda x: x
        coeffs = basis.project(f)

        # x = L_0 - L_1, so coefficients should be [1, -1, 0, ...]
        self.assertAlmostEqual(coeffs[0], 1.0, places=8)
        self.assertAlmostEqual(coeffs[1], -1.0, places=8)

    def test_approximation_reconstruction(self):
        """Test that approximation reconstructs the function."""
        basis = LaguerreBasis(20)
        f = lambda x: math.exp(-x/2)  # Function decaying on [0, inf)

        approx = basis.approximate(f)

        # Test at several points
        for x in [0.0, 0.5, 1.0, 2.0, 5.0]:
            f_val = f(x)
            approx_val = approx(x)
            self.assertAlmostEqual(f_val, approx_val, places=6,
                msg=f"Approximation at x={x}")

    def test_function_projection_utility(self):
        """Test function_projection utility."""
        f = lambda x: math.exp(-x/2)
        coeffs = function_projection(f, max_degree=10)
        self.assertEqual(len(coeffs), 11)

    def test_function_approximation_utility(self):
        """Test function_approximation utility."""
        f = lambda x: math.exp(-x/2)
        approx = function_approximation(f, max_degree=10)

        # Test reconstruction
        self.assertAlmostEqual(approx(1.0), f(1.0), places=6)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_to_sympy_standard(self):
        """Test conversion to SymPy Laguerre polynomial."""
        try:
            import sympy as sp
            x = sp.Symbol("x")

            for n in range(5):
                expr = to_sympy(n, alpha=0)
                expected = sp.laguerre(n, x)
                self.assertEqual(expand(expr), expand(expected))
        except ImportError:
            self.skipTest("SymPy not available")

    def test_to_sympy_generalized(self):
        """Test conversion to SymPy generalized Laguerre polynomial."""
        try:
            import sympy as sp
            x = sp.Symbol("x")

            for n in range(4):
                for alpha in [0.5, 1.0]:
                    expr = to_sympy(n, alpha=alpha)
                    expected = sp.assoc_laguerre(n, alpha, x)
                    self.assertEqual(expand(expr), expand(expected))
        except ImportError:
            self.skipTest("SymPy not available")

    def test_evaluate_array(self):
        """Test array evaluation."""
        x_values = np.linspace(0, 5, 10)
        result = evaluate_array(3, x_values)

        self.assertEqual(len(result), 10)
        # Compare with scalar evaluation
        L3 = LaguerrePolynomial(3)
        for i, x in enumerate(x_values):
            expected = L3.evaluate(x)
            self.assertAlmostEqual(result[i], expected, places=14)

    def test_evaluate_array_generalized(self):
        """Test array evaluation with alpha."""
        x_values = np.linspace(0, 5, 10)
        result = evaluate_array(3, x_values, alpha=1.0)

        self.assertEqual(len(result), 10)
        L3_gen = GeneralizedLaguerrePolynomial(3, alpha=1.0)
        for i, x in enumerate(x_values):
            expected = L3_gen.evaluate(x)
            self.assertAlmostEqual(result[i], expected, places=14)

    def test_generate_basis_matrix(self):
        """Test basis matrix generation."""
        x_values = np.linspace(0, 5, 20)
        V = generate_basis_matrix(5, x_values)

        self.assertEqual(V.shape, (20, 6))  # 20 points, degrees 0-5

        # First column should be all ones (L_0 = 1)
        np.testing.assert_array_almost_equal(V[:, 0], np.ones(20), decimal=14)

    def test_evaluate_mp(self):
        """Test high-precision evaluation."""
        result = evaluate_mp(5, 2.0, alpha=0, prec=50)

        # Compare with standard evaluation
        L5 = LaguerrePolynomial(5)
        expected = L5.evaluate(2.0)
        self.assertAlmostEqual(float(result), expected, places=14)

    def test_stable_evaluation(self):
        """Test stable evaluation function."""
        result_float = stable_evaluation(5, 2.0, alpha=0, use_mp=False)
        result_mp = stable_evaluation(5, 2.0, alpha=0, use_mp=True)

        self.assertAlmostEqual(float(result_float), float(result_mp), places=14)

    def test_condition_estimate(self):
        """Test condition estimate function."""
        cond = condition_estimate(10, 5.0, alpha=0)
        self.assertGreater(cond, 0)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling."""

    def test_negative_degree_laguerre(self):
        """Test negative degree raises ValueError."""
        with self.assertRaises(ValueError):
            LaguerrePolynomial(-1)

    def test_negative_degree_generalized(self):
        """Test negative degree raises ValueError for generalized."""
        with self.assertRaises(ValueError):
            GeneralizedLaguerrePolynomial(-1, alpha=0.5)

    def test_invalid_alpha(self):
        """Test invalid alpha raises ValueError."""
        with self.assertRaises(ValueError):
            GeneralizedLaguerrePolynomial(5, alpha=-1)
        with self.assertRaises(ValueError):
            GeneralizedLaguerrePolynomial(5, alpha=-2)

    def test_negative_max_degree_basis(self):
        """Test negative max_degree raises ValueError."""
        with self.assertRaises(ValueError):
            LaguerreBasis(-1)
        with self.assertRaises(ValueError):
            GeneralizedLaguerreBasis(-1, alpha=0.5)


class TestMathematicalProperties(unittest.TestCase):
    """Tests for key mathematical properties."""

    def test_l_n_at_zero(self):
        """Test L_n(0) = 1 for all n."""
        for n in range(30):
            Ln = LaguerrePolynomial(n)
            value = Ln.evaluate(0)
            self.assertAlmostEqual(value, 1.0, places=14,
                msg=f"L_{n}(0) should equal 1")

    def test_l_n_at_one(self):
        """Test L_n(1) values."""
        # Known: sum of coefficients gives value at x=1
        for n in range(10):
            Ln = LaguerrePolynomial(n)
            coeffs = Ln.get_coefficients()
            value_at_one = Ln.evaluate(1)
            sum_coeffs = sum(coeffs)
            self.assertAlmostEqual(value_at_one, sum_coeffs, places=14,
                msg=f"L_{n}(1) should equal sum of coefficients")

    def test_alternating_signs(self):
        """Test that L_n(x) has n sign changes in (0, inf)."""
        # This is a property but hard to test precisely
        # Just verify roots are positive and distinct
        for n in range(1, 10):
            roots = compute_roots(n)
            self.assertEqual(len(roots), n)
            for r in roots:
                self.assertGreater(r, 0, msg=f"Root of L_{n} should be positive")

    def test_derivatives_at_zero(self):
        """Test derivatives at x=0."""
        # L_n'(0) = -n for standard Laguerre
        for n in range(1, 10):
            Ln = LaguerrePolynomial(n)
            h = 1e-8
            deriv_approx = (Ln.evaluate(h) - Ln.evaluate(-h)) / (2*h)
            # Note: this is tricky since domain is [0, inf), use forward difference
            deriv_forward = (Ln.evaluate(2*h) - Ln.evaluate(h)) / h
            self.assertAlmostEqual(deriv_forward, -n, places=6,
                msg=f"L_{n}'(0) should equal {-n}")


if __name__ == "__main__":
    unittest.main()
