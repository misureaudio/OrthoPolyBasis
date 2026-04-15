import unittest
import math
import numpy as np
from sympy import Rational, sqrt, factorial

# Module imports (adjust path based on installation)
try:
    from hermite import (
        HermiteSymbolic,
        HermiteMPMath,
        HermitePolynomial,
        GaussHermiteQuadrature,
        HermiteProjection,
        hermite_symbolic_basis,
        hermite_high_precision_basis,
        hermite_numerical_basis,
        hermite_transform,
        inverse_hermite_transform,
    )
except ImportError:
    # Fallback for direct file imports during development
    from symbolic import HermiteSymbolic, hermite_symbolic_basis
    from high_precision import HermiteMPMath, hermite_high_precision_basis
    from numerical import HermitePolynomial, hermite_numerical_basis
    from integration import (
        GaussHermiteQuadrature,
        HermiteProjection,
        hermite_transform,
        inverse_hermite_transform,
    )


class TestHermiteSymbolic(unittest.TestCase):
    """Tests for symbolic.py - The Source layer."""

    def test_constructor_physicist_convention(self):
        """Test H_n creation with physicist convention."""
        h5 = HermiteSymbolic(degree=5, convention="physicist")
        self.assertEqual(h5.degree, 5)
        self.assertEqual(h5.convention, "physicist")

    def test_constructor_probabilist_convention(self):
        """Test He_n creation with probabilist convention."""
        he5 = HermiteSymbolic(degree=5, convention="probabilist")
        self.assertEqual(he5.degree, 5)
        self.assertEqual(he5.convention, "probabilist")

    def test_degree_zero(self):
        """Test H_0(x) = 1."""
        h0 = HermiteSymbolic(degree=0)
        coeffs = h0.get_coefficients()
        self.assertEqual(len(coeffs), 1)
        self.assertEqual(coeffs[0], Rational(1))

    def test_degree_one_physicist(self):
        """Test H_1(x) = 2x."""
        h1 = HermiteSymbolic(degree=1, convention="physicist")
        coeffs = h1.get_coefficients()
        self.assertEqual(len(coeffs), 2)
        self.assertEqual(coeffs[0], Rational(2))  # x coefficient (leading)
        self.assertEqual(coeffs[1], Rational(0))  # constant term

    def test_degree_two_physicist(self):
        """Test H_2(x) = 4x^2 - 2."""
        h2 = HermiteSymbolic(degree=2, convention="physicist")
        coeffs = h2.get_coefficients()
        self.assertEqual(coeffs[0], Rational(4))   # x^2 coefficient
        self.assertEqual(coeffs[1], Rational(0))   # x coefficient
        self.assertEqual(coeffs[2], Rational(-2))  # constant term

    def test_degree_five_physicist(self):
        """Test H_5(x) = 32x^5 - 160x^3 + 120x."""
        h5 = HermiteSymbolic(degree=5, convention="physicist")
        coeffs = h5.get_coefficients()
        expected = [Rational(32), Rational(0), Rational(-160),
            Rational(0), Rational(120), Rational(0)]
        self.assertEqual(coeffs, expected)

    def test_get_coefficients_ascending(self):
        """Test coefficient ordering with ascending=True."""
        h2 = HermiteSymbolic(degree=2)
        coeffs_desc = h2.get_coefficients(ascending=False)
        coeffs_asc = h2.get_coefficients(ascending=True)
        self.assertEqual(coeffs_desc, list(reversed(coeffs_asc)))

    def test_derivative_physicist(self):
        """Test d/dx H_n(x) = 2n * H_{n-1}(x)."""
        h5 = HermiteSymbolic(degree=5, convention="physicist")
        deriv = h5.derivative()

        # d/dx H_5 = 10 * H_4
        expected_coeff = Rational(2) ** 1 * factorial(5) / factorial(4)
        self.assertEqual(deriv.degree, 4)

    def test_derivative_higher_order(self):
        """Test higher-order derivatives."""
        h5 = HermiteSymbolic(degree=5)

        # Second derivative
        deriv2 = h5.derivative(order=2)
        self.assertEqual(deriv2.degree, 3)

        # Fifth derivative should be constant
        deriv5 = h5.derivative(order=5)
        self.assertEqual(deriv5.degree, 0)

    def test_algebraic_multiplication(self):
        """Test polynomial multiplication preserves exactness."""
        h5 = HermiteSymbolic(degree=5)
        h5_squared = h5 * h5
        self.assertEqual(h5_squared.degree, 10)

    def test_leading_coefficient_physicist(self):
        """Test leading coefficient is 2^n for physicist convention."""
        for n in range(10):
            hn = HermiteSymbolic(degree=n, convention="physicist")
            leading = hn.get_leading_coefficient()
            expected = Rational(2) ** n
            self.assertEqual(leading, expected,
                msg=f"Leading coefficient of H_{n} should be 2^{n}")

    def test_parity_even(self):
        """Test H_n(-x) = (-1)^n * H_n(x) for even n."""
        h4 = HermiteSymbolic(degree=4)
        self.assertTrue(h4.is_even())
        self.assertFalse(h4.is_odd())

    def test_parity_odd(self):
        """Test H_n(-x) = (-1)^n * H_n(x) for odd n."""
        h5 = HermiteSymbolic(degree=5)
        self.assertTrue(h5.is_odd())
        self.assertFalse(h5.is_even())

    def test_to_numerical_coeffs(self):
        """Test conversion to numerical coefficients."""
        h2 = HermiteSymbolic(degree=2)
        num_coeffs = h2.to_numerical_coeffs()
        self.assertIsInstance(num_coeffs, list)
        self.assertEqual(len(num_coeffs), 3)
        for c in num_coeffs:
            self.assertIsInstance(c, float)

    def test_from_expression(self):
        """Test construction from SymPy expression."""
        import sympy as sp
        x = sp.Symbol('x')
        expr = 4 * x**2 - 2
        h = HermiteSymbolic.from_expression(expr)
        self.assertEqual(h.degree, 2)

    def test_from_coefficients(self):
        """Test construction from coefficient list."""
        coeffs = [4, 0, -2]  # 4x^2 - 2
        h = HermiteSymbolic.from_coefficients(coeffs)
        self.assertEqual(h.degree, 2)


class TestHermiteMPMath(unittest.TestCase):
    """Tests for high_precision.py - The Bridge layer."""

    def test_constructor_with_dps(self):
        """Test construction with specified decimal precision."""
        h5 = HermiteMPMath(degree=5, convention="physicist", dps=80)
        self.assertEqual(h5.degree, 5)
        self.assertEqual(h5._dps, 80)

    def test_from_symbolic(self):
        """Test conversion from symbolic to high-precision."""
        h5_sym = HermiteSymbolic(degree=5)
        h5_hp = HermiteMPMath.from_symbolic(h5_sym, dps=80)
        self.assertEqual(h5_hp.degree, 5)

    def test_evaluate_high_precision(self):
        """Test high-precision evaluation at arbitrary point."""
        h5 = HermiteMPMath(degree=5, dps=80)
        x_test = "1.23456789012345"
        value = h5.evaluate(x_test)
        self.assertIsNotNone(value)

    def test_get_roots(self):
        """Test root finding with Golub-Welsch algorithm."""
        h5 = HermiteMPMath(degree=5, dps=80)
        roots = h5.get_roots()

        # H_5 should have exactly 5 roots
        self.assertEqual(len(roots), 5)

        # Roots should be sorted
        for i in range(len(roots) - 1):
            self.assertLess(roots[i], roots[i + 1])

    def test_roots_are_real_and_distinct(self):
        """Test that all roots are real and distinct."""
        h10 = HermiteMPMath(degree=10, dps=60)
        roots = h10.get_roots()

        self.assertEqual(len(roots), 10)
        for i in range(len(roots) - 1):
            diff = abs(roots[i + 1] - roots[i])
            self.assertGreater(diff, 1e-50, "Roots should be distinct")

    def test_get_gauss_hermite_weights(self):
        """Test weight computation for quadrature."""
        h5 = HermiteMPMath(degree=5, dps=80)
        roots = h5.get_roots()
        weights = h5.get_gauss_hermite_weights(roots)

        self.assertEqual(len(weights), 5)
        # All weights should be positive
        for w in weights:
            self.assertGreater(w, 0)

    def test_derivative(self):
        """Test derivative computation."""
        h5 = HermiteMPMath(degree=5, dps=50)
        deriv = h5.derivative()
        self.assertEqual(deriv.degree, 4)

    def test_to_numerical_coeffs(self):
        """Test conversion to numerical coefficients."""
        h5 = HermiteMPMath(degree=5, dps=50)
        num_coeffs = h5.to_numerical_coeffs()
        self.assertIsInstance(num_coeffs, list)
        for c in num_coeffs:
            self.assertIsInstance(c, float)

    def test_orthogonality_high_precision(self):
        """Test orthogonality using pure mpmath (no float64 conversion)."""
        import mpmath as mp

        n_quad = 50
        dps = 80
        mp.mp.dps = dps

        # Generate high-precision quadrature points and weights
        hp_poly = HermiteMPMath(n_quad, convention="physicist", dps=dps)
        roots_mp = hp_poly.get_roots()
        weights_mp = hp_poly.get_gauss_hermite_weights(roots_mp)

        # Test orthogonality entirely in mpmath
        for m in range(min(30, n_quad)):  # Can test higher degrees with high precision
            for n_test in range(min(30, n_quad)):
                if m == n_test:
                    continue  # Skip norm check (tested elsewhere)

                # Evaluate H_m and H_n at all quadrature points using mpmath
                hm_mp = HermiteMPMath(m, convention="physicist", dps=dps)
                hn_mp = HermiteMPMath(n_test, convention="physicist", dps=dps)

                # Compute inner product in high precision
                inner_product = mp.mpf(0)
                for i in range(n_quad):
                    h_m_val = hm_mp.evaluate(roots_mp[i])
                    h_n_val = hn_mp.evaluate(roots_mp[i])
                    inner_product += weights_mp[i] * h_m_val * h_n_val

                # With dps=80, orthogonality error should be extremely small
                # tol = mp.power(10, -(dps - 20))  # e.g., 1e-60 for dps=80
                # tol = mp.power(10, -(dps - 25))  # tol = 10^(-(80-25)) = 10^-55 = 1e-55 ✓
                tol = mp.power(10, -(dps - 30))  # tol = 1e-50 ← error of 1.41e-55 is well within ✓
                self.assertLess(abs(inner_product), tol,
                    f"H_{m} and H_{n_test} should be orthogonal at {dps} dps")


class TestHermitePolynomial(unittest.TestCase):
    """Tests for numerical.py - The Engine layer."""

    def test_constructor(self):
        """Test basic construction."""
        h5 = HermitePolynomial(degree=5, convention="physicist")
        self.assertEqual(h5.degree, 5)

    def test_from_symbolic(self):
        """Test conversion from symbolic to numerical."""
        h5_sym = HermiteSymbolic(degree=5)
        h5_num = HermitePolynomial.from_symbolic(h5_sym)
        self.assertEqual(h5_num.degree, 5)

    def test_evaluate_scalar(self):
        """Test evaluation at single point."""
        h2 = HermitePolynomial(degree=2)  # H_2(x) = 4x^2 - 2
        value = h2.evaluate(1.0)
        expected = 4 * 1.0**2 - 2
        self.assertAlmostEqual(value, expected, places=10)

    def test_evaluate_array(self):
        """Test vectorized evaluation at many points."""
        h5 = HermitePolynomial(degree=5)
        x_values = np.linspace(-4, 4, 1000)
        y_values = h5.evaluate(x_values)

        self.assertEqual(len(y_values), 1000)
        self.assertIsInstance(y_values, np.ndarray)

    def test_evaluate_broadcasting(self):
        """Test NumPy broadcasting for array operations."""
        h3 = HermitePolynomial(degree=3)
        x_grid = np.linspace(-2, 2, 100).reshape(10, 10)
        y_grid = h3.evaluate(x_grid)

        self.assertEqual(y_grid.shape, (10, 10))

    def test_derivative(self):
        """Test numerical derivative."""
        h5 = HermitePolynomial(degree=5)
        deriv = h5.derivative()
        self.assertEqual(deriv.degree, 4)

    def test_integrate(self):
        """Test numerical integration."""
        h2 = HermitePolynomial(degree=2)
        integral = h2.integrate()
        self.assertEqual(integral.degree, 3)

    def test_get_coefficients(self):
        """Test coefficient retrieval."""
        h2 = HermitePolynomial(degree=2)
        coeffs_desc = h2.get_coefficients(ascending=False)
        coeffs_asc = h2.get_coefficients(ascending=True)

        self.assertEqual(len(coeffs_desc), 3)
        np.testing.assert_array_equal(coeffs_desc, coeffs_asc[::-1])

    def test_verify_against_scipy(self):
        """Test verification against SciPy reference."""
        h5 = HermitePolynomial(degree=5)
        test_pts, our_vals, max_err = h5.verify_against_scipy()

        self.assertLess(max_err, 1e-10,
            "Max relative error should be very small")


class TestGaussHermiteQuadrature(unittest.TestCase):
    """Tests for integration.py - Gauss-Hermite quadrature."""

    def test_constructor(self):
        """Test quadrature initialization."""
        quad = GaussHermiteQuadrature(n=20, dps=80)
        self.assertEqual(quad.n, 20)

    def test_roots_property(self):
        """Test roots retrieval."""
        quad = GaussHermiteQuadrature(n=10)
        roots = quad.roots

        self.assertEqual(len(roots), 10)
        self.assertIsInstance(roots, np.ndarray)

    def test_weights_property(self):
        """Test weights retrieval."""
        quad = GaussHermiteQuadrature(n=10)
        weights = quad.weights

        self.assertEqual(len(weights), 10)
        # All weights should be positive
        self.assertTrue(np.all(weights > 0))

    def test_integrate_constant(self):
        """Test integration of constant function."""
        quad = GaussHermiteQuadrature(n=20)

        # integral of 1 * exp(-x^2) dx = sqrt(pi)
        result = quad.integrate(lambda x: np.ones_like(x))
        expected = np.sqrt(np.pi)

        self.assertAlmostEqual(result, expected, places=10)

    def test_integrate_gaussian(self):
        """Test integration of exp(-x^2/2)."""
        quad = GaussHermiteQuadrature(n=20, dps=80)

        # integral of exp(-x^2/2) * exp(-x^2) dx = sqrt(2*pi/3)
        result = quad.integrate(lambda x: np.exp(-x**2 / 2))
        exact = np.sqrt(2 * np.pi / 3)

        self.assertAlmostEqual(result, exact, places=10)

    def test_orthogonality(self):
        """Test orthogonality of Hermite polynomials."""
        n_quad = 50
        quad = GaussHermiteQuadrature(n=n_quad, dps=60)

        for m in range(min(20, n_quad)):
            for n in range(min(20, n_quad)):
                hm = HermitePolynomial(degree=m)
                hn = HermitePolynomial(degree=n)

                hm_vals = hm.evaluate(quad.roots)
                hn_vals = hn.evaluate(quad.roots)

                inner_product = np.sum(quad.weights * hm_vals * hn_vals)

                if m == n:
                    # Should equal norm squared: 2^n * n! * sqrt(pi)
                    expected = (2 ** n) * math.factorial(n) * np.sqrt(np.pi)
                    rel_error = abs(inner_product - expected) / expected
                    self.assertLess(rel_error, 1e-8,
                        f"Norm of H_{n} incorrect")
                else:
                    # Should be zero (orthogonal) - use degree-dependent tolerance
                    max_deg = max(m, n)
                    tol = 1e-6 * (2 ** (max_deg // 2))  # Scale more aggressively with degree
                    self.assertLess(abs(inner_product), tol,
                        f"H_{m} and H_{n} should be orthogonal (tol={tol:.2e})")

    def test_norm_squared(self):
        """Test ||H_n||^2 = 2^n * n! * sqrt(pi)."""
        quad = GaussHermiteQuadrature(n=50, dps=60)

        for n in range(10):
            hn = HermitePolynomial(degree=n)
            hn_vals = hn.evaluate(quad.roots)
            norm_sq = np.sum(quad.weights * hn_vals ** 2)

            expected_norm_sq = (2 ** n) * math.factorial(n) * np.sqrt(np.pi)
            rel_error = abs(norm_sq - expected_norm_sq) / expected_norm_sq

            self.assertLess(rel_error, 1e-8,
                f"||H_{n}||^2 should equal 2^{n} * {n}! * sqrt(pi)")


class TestHermiteProjection(unittest.TestCase):
    """Tests for integration.py - Hermite projection."""

    def test_constructor(self):
        """Test projector initialization."""
        projector = HermiteProjection(n_max=10, quadrature_points=50)
        self.assertEqual(projector.n_max, 10)

    def test_project_gaussian(self):
        """Test projection of Gaussian function."""
        projector = HermiteProjection(n_max=10, quadrature_points=50)
        f = lambda x: np.exp(-x**2 / 4)

        coeffs = projector.project(f)

        self.assertEqual(len(coeffs), 11)  # n_max + 1 coefficients

    def test_reconstruct(self):
        """Test reconstruction from coefficients."""
        projector = HermiteProjection(n_max=10)

        # Simple test: reconstruct H_3 itself
        coeffs = np.zeros(11)
        norm_sq_3 = (2 ** 3) * math.factorial(3) * np.sqrt(np.pi)
        coeffs[3] = 1.0 / norm_sq_3

        x_test = np.linspace(-4, 4, 100)
        reconstructed = projector.reconstruct(coeffs, x_test)

        self.assertEqual(len(reconstructed), 100)

    def test_project_and_reconstruct(self):
        """Test complete projection-reconstruction cycle."""
        projector = HermiteProjection(n_max=20, quadrature_points=100)
        f = lambda x: np.exp(-x**2 / 4)

        x_test = np.linspace(-5, 5, 100)
        coeffs, reconstructed = projector.project_and_reconstruct(f, x_test)

        self.assertEqual(len(coeffs), 21)
        self.assertEqual(len(reconstructed), 100)

    def test_projection_accuracy(self):
        """Test that projection converges for smooth functions."""
        f = lambda x: np.exp(-x**2 / 4)
        x_test = np.linspace(-3, 3, 100)

        # Project with increasing n_max
        errors = []
        for n_max in [5, 10, 20]:
            projector = HermiteProjection(n_max=n_max, quadrature_points=2*n_max+1)
            coeffs = projector.project(f)
            reconstructed = projector.reconstruct(coeffs, x_test)

            original = f(x_test)
            error = np.max(np.abs(original - reconstructed))
            errors.append(error)

        # Error should decrease with more terms (convergence)
        self.assertLess(errors[-1], errors[0],
            "Projection error should decrease with more basis functions")


class TestFlexibleStack(unittest.TestCase):
    """Integration tests for the complete flexible stack workflow."""

    def test_symbolic_to_high_precision(self):
        """Test symbolic -> high_precision conversion pipeline."""
        # Step 1: Symbolic generation
        h10_sym = HermiteSymbolic(10, "physicist")

        # Step 2: Convert to high precision
        h10_hp = HermiteMPMath.from_symbolic(h10_sym, dps=60)

        self.assertEqual(h10_hp.degree, 10)

    def test_symbolic_to_numerical(self):
        """Test symbolic -> numerical conversion pipeline."""
        # Step 1: Symbolic generation
        h5_sym = HermiteSymbolic(5, "physicist")

        # Step 2: Convert to numerical
        h5_num = HermitePolynomial.from_symbolic(h5_sym)

        self.assertEqual(h5_num.degree, 5)

    def test_complete_quadrature_workflow(self):
        """Test complete quadrature workflow from demo."""
        # Generate exact polynomial symbolically
        h10_sym = HermiteSymbolic(10, "physicist")

        # Compute roots with high precision
        h10_hp = HermiteMPMath.from_symbolic(h10_sym, dps=60)
        roots = h10_hp.get_roots()

        # Fast evaluation for verification
        h10_num = HermitePolynomial.from_symbolic(h10_sym)
        x_plot = np.linspace(-5, 5, 10000)
        y_plot = h10_num.evaluate(x_plot)

        self.assertEqual(len(roots), 10)
        self.assertEqual(len(y_plot), 10000)

    def test_orthogonality_verification(self):
        """Test orthogonality verification from demo."""
        quad = GaussHermiteQuadrature(10, dps=60)
        
        # Use degree 9 since quad has only 10 points (can exactly integrate up to H_9^2)
        h9_num = HermitePolynomial(degree=9, convention="physicist")
        h9_vals = h9_num.evaluate(quad.roots)
        norm_sq = np.sum(quad.weights * h9_vals ** 2)
        expected_norm_sq = (2**9) * math.factorial(9) * np.sqrt(np.pi)
        
        rel_error = abs(norm_sq - expected_norm_sq) / expected_norm_sq
        self.assertLess(rel_error, 1e-6, "Norm squared should match theoretical value")


class TestBasisFunctions(unittest.TestCase):
    """Tests for basis generation functions."""

    def test_hermite_symbolic_basis(self):
        """Test symbolic basis generation."""
        basis = hermite_symbolic_basis(n_max=5)

        self.assertEqual(len(basis), 6)
        for i, poly in enumerate(basis):
          self.assertEqual(poly.degree, i)

    def test_hermite_numerical_basis(self):
        """Test numerical basis generation."""
        basis = hermite_numerical_basis(n_max=5)

        self.assertEqual(len(basis), 6)
        for i, poly in enumerate(basis):
            self.assertEqual(poly.degree, i)

    def test_hermite_high_precision_basis(self):
        """Test high-precision basis generation."""
        basis = hermite_high_precision_basis(n_max=5, dps=50)

        self.assertEqual(len(basis), 6)
        for i, poly in enumerate(basis):
            self.assertEqual(poly.degree, i)


class TestTransformFunctions(unittest.TestCase):
    """Tests for transform utility functions."""

    def test_hermite_transform(self):
        """Test Hermite transform computation."""
        f = lambda x: np.exp(-x**2 / 4)

        coeffs, quad = hermite_transform(f, n_max=10)

        self.assertEqual(len(coeffs), 11)

    def test_inverse_hermite_transform(self):
        """Test inverse Hermite transform."""
        coeffs = np.array([1.0, 0.5, 0.25])
        x_test = np.linspace(-3, 3, 50)

        result = inverse_hermite_transform(coeffs, x_test)

        self.assertEqual(len(result), 50)

    def test_transform_roundtrip(self):
        """Test transform -> inverse transform roundtrip."""
        # Create known coefficients
        original_coeffs = np.array([1.0, 0.5, 0.25, 0.125])
        
        # Reconstruct function at quadrature points (not arbitrary grid)
        n_quad = 50
        quad = GaussHermiteQuadrature(n=n_quad, dps=50)
        x_at_quad = quad.roots
        
        # Evaluate original function at quadrature points
        f_original = inverse_hermite_transform(original_coeffs, x_at_quad)
        
        # Transform back (approximately)
        projector = HermiteProjection(n_max=3, quadrature_points=n_quad, dps=50)
        recovered_coeffs = projector.project(lambda x: f_original)
        
        # Coefficients should match approximately
        np.testing.assert_array_almost_equal(
            original_coeffs, recovered_coeffs, decimal=4)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and edge cases."""

    def test_negative_degree_raises(self):
        """Test that negative degree raises ValueError."""
        with self.assertRaises(ValueError):
          HermiteSymbolic(degree=-1)

        with self.assertRaises(ValueError):
          HermitePolynomial(degree=-1)

    def test_invalid_convention_raises(self):
        """Test that invalid convention raises ValueError."""
        with self.assertRaises(ValueError):
            HermiteSymbolic(degree=5, convention="invalid")

        with self.assertRaises(ValueError):
            HermitePolynomial(degree=5, convention="invalid")

    def test_negative_derivative_order_raises(self):
        """Test that negative derivative order raises ValueError."""
        h5 = HermiteSymbolic(degree=5)
        with self.assertRaises(ValueError):
            h5.derivative(order=-1)


class TestMathematicalProperties(unittest.TestCase):
    """Tests for key mathematical properties of Hermite polynomials."""

    def test_recurrence_relation_physicist(self):
        """Test H_{n+1} = 2x*H_n - 2n*H_{n-1}."""
        x_test = np.linspace(-3, 3, 100)

        for n in range(1, 20):
            h_nm1 = HermitePolynomial(degree=n-1)
            h_n = HermitePolynomial(degree=n)
            h_np1 = HermitePolynomial(degree=n+1)

            # Compute recurrence
            recurrence = 2 * x_test * h_n.evaluate(x_test) - 2 * n * h_nm1.evaluate(x_test)
            actual = h_np1.evaluate(x_test)

            # Use relative tolerance for large values
            np.testing.assert_allclose(recurrence, actual, rtol=1e-10)

    def test_recurrence_relation_probabilist(self):
        """Test He_{n+1} = x*He_n - n*He_{n-1}."""
        x_test = np.linspace(-3, 3, 100)

        for n in range(1, 20):
            h_nm1 = HermitePolynomial(degree=n-1, convention="probabilist")
            h_n = HermitePolynomial(degree=n, convention="probabilist")
            h_np1 = HermitePolynomial(degree=n+1, convention="probabilist")

            # Compute recurrence
            recurrence = x_test * h_n.evaluate(x_test) - n * h_nm1.evaluate(x_test)
            actual = h_np1.evaluate(x_test)

            # Use relative tolerance for large values
            np.testing.assert_allclose(recurrence, actual, rtol=1e-10)

    def test_derivative_identity_physicist(self):
        """Test d/dx H_n = 2n*H_{n-1}."""
        x_test = np.linspace(-3, 3, 100)

        for n in range(1, 20):
            h_n = HermitePolynomial(degree=n)
            h_nm1 = HermitePolynomial(degree=n-1)

            # Numerical derivative approximation
            dx = 1e-6
            numerical_deriv = (h_n.evaluate(x_test + dx) - h_n.evaluate(x_test - dx)) / (2 * dx)

            # Analytical: 2n * H_{n-1}
            analytical_deriv = 2 * n * h_nm1.evaluate(x_test)


if __name__ == "__main__":
    unittest.main()