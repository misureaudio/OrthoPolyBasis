"""Hermite Polynomial Integration Module - The Client.

This module provides Gaussian quadrature and projection capabilities,
serving as the client that orchestrates the three-layer stack:
    - high_precision.py: For computing roots and weights (critical for accuracy)
    - numerical.py: For fast function evaluation at quadrature points
    - symbolic.py: For exact coefficient manipulation when needed
"""

from __future__ import annotations
from typing import Callable, Optional, Union, List, Tuple, Any
import numpy as np
import mpmath as mp

import math
import warnings

# Import from our modules
try:
    from .high_precision import HermiteMPMath, hermite_high_precision_basis
    FROM_PACKAGE = True
except ImportError:
    from high_precision import HermiteMPMath, hermite_high_precision_basis
    FROM_PACKAGE = False

try:
    from .numerical import HermitePolynomial, hermite_numerical_basis
    NUM_FROM_PACKAGE = True
except ImportError:
    from numerical import HermitePolynomial, hermite_numerical_basis
    NUM_FROM_PACKAGE = False


# Module-level cache for quadrature roots and weights
# Key: (n, dps, use_mpmath), Value: (roots_array, weights_array)
_gauss_hermite_cache = {}



class GaussHermiteQuadrature:
    """Gauss-Hermite quadrature for integrals with weight exp(-x^2).

    Computes approximations of integrals of the form:
        integral_{-inf}^{+inf} f(x) * exp(-x^2) dx ~= sum_{i=1}^n w_i * f(x_i)

    Uses high_precision.py for root/weight computation (critical for n > 50)
    and numerical.py for fast evaluation of the integrand.
    """

    def __init__(self, n: int, dps: int = 50, use_mpmath: bool = True):
        """Initialize quadrature with n points.

        Args:
            n: Number of quadrature points.
            dps: Decimal precision for root/weight computation (default: 50).
                 Only used when use_mpmath=True.
            use_mpmath: If True, use high-precision mpmath library (accurate but slow).
                       If False, use NumPy Golub-Welsch algorithm (fast, double precision).
                       Default: True for backward compatibility and accuracy at high n.

        Notes:
            - use_mpmath=False is recommended for n < 100 where double precision suffices
            - use_mpmath=True is recommended for n >= 100 or when extreme accuracy needed
        """
        self.n = n
        self._dps = dps
        self._use_mpmath = use_mpmath

        # Warn if using NumPy at high degrees
        if not use_mpmath and n >= 100:
            warnings.warn(
                f"Hermite quadrature with n={n} using NumPy (double precision). "
                "For n >= 100, consider use_mpmath=True for better accuracy. "
                "NumPy may lose precision due to high condition numbers.",
                UserWarning,
                stacklevel=2
            )

        # Check module-level cache first (key includes use_mpmath flag)
        cache_key = (n, dps, use_mpmath)
        if cache_key not in _gauss_hermite_cache:
            if use_mpmath:
                # High-precision path using mpmath
                hp_poly = HermiteMPMath(n, convention="physicist", dps=dps)
                mp_roots = hp_poly.get_roots()
                mp_weights = hp_poly.get_gauss_hermite_weights(mp_roots)

                # Convert to NumPy for fast evaluation and cache
                _gauss_hermite_cache[cache_key] = (
                    np.array([float(r) for r in mp_roots]),
                    np.array([float(w) for w in mp_weights])
                )
            else:
                # Fast path using NumPy Golub-Welsch algorithm
                _gauss_hermite_cache[cache_key] = self._compute_golub_welsch(n)

        # Load from cache (share the arrays, no copy needed)
        self._roots, self._weights = _gauss_hermite_cache[cache_key]

    def _compute_golub_welsch(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Gauss-Hermite nodes and weights using Golub-Welsch algorithm.

        Uses NumPy double precision - fast but limited to ~16 digits accuracy.
        Suitable for n < 100 where condition numbers are manageable.

        Returns:
            Tuple of (roots, weights) as NumPy arrays
        """
        # Build Jacobi matrix for Hermite polynomials
        # Diagonal elements: all zeros for Hermite
        # Off-diagonal: sqrt(k/2) for k = 1, 2, ..., n-1

        diag = np.zeros(n)
        # off_diag = np.sqrt(np.arange(1, n) / 2.0)
        off_diag = np.sqrt(np.arange(1, n))

        # Construct symmetric tridiagonal Jacobi matrix
        J = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

        # Eigenvalues are the quadrature nodes (roots)
        eigenvalues, eigenvectors = np.linalg.eigh(J)

        # Weights from first row of eigenvector matrix
        # weights = 2.0 * (eigenvectors[0, :] ** 2)
        weights = np.sqrt(math.pi) * (eigenvectors[0, :] ** 2)

        return eigenvalues, weights

    @property
    def roots(self) -> np.ndarray:
        """Quadrature points (roots of H_n)."""
        return self._roots.copy()

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights."""
        return self._weights.copy()

    def integrate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        vectorized: bool = True
    ) -> float:
        """Compute integral of f(x) * exp(-x^2) from -inf to +inf.

        Args:
            f: Function to integrate (should accept NumPy arrays).
            vectorized: If True, f accepts array input; if False, f is called
                       point-by-point.

        Returns:
            Approximate value of the integral.
        """
        if vectorized:
            f_values = f(self._roots)
        else:
            f_values = np.array([f(x) for x in self._roots])

        return float(np.sum(self._weights * f_values))

    def integrate_with_weight(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        custom_weight: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> float:
        """Compute integral with optional additional weight function.

        Computes: integral f(x) * exp(-x^2) * w(x) dx

        Args:
            f: Function to integrate.
            custom_weight: Additional weight function (default: 1).
        """
        if custom_weight is None:
            return self.integrate(f)

        def weighted_f(x):
            return f(x) * custom_weight(x)

        return self.integrate(weighted_f)


class HermiteProjection:
    """Project functions onto the Hermite polynomial basis.

    Computes coefficients c_n in the expansion:
        f(x) ~= sum_{n=0}^{N} c_n * H_n(x)

    where c_n = <f, H_n> / <H_n, H_n>
    """

    def __init__(self, n_max: int, quadrature_points: Optional[int] = None, dps: int = 50):
        """Initialize projector.

        Args:
            n_max: Maximum degree of basis polynomials.
            quadrature_points: Number of quadrature points (default: 2*n_max + 1).
            dps: Precision for quadrature computation.
        """
        self.n_max = n_max
        self._dps = dps

        # Use enough quadrature points to integrate products exactly
        if quadrature_points is None:
            quadrature_points = max(2 * n_max + 1, 50)

        self._quadrature = GaussHermiteQuadrature(quadrature_points, dps=dps)

        # Precompute basis polynomials and norms
        self._basis = hermite_numerical_basis(n_max, "physicist")

        # Norm squared: <H_n, H_n> = 2^n * n! * sqrt(pi)
        self._norms_sq = np.array([
            2**n * math.factorial(n) * np.sqrt(np.pi)
            for n in range(n_max + 1)
        ])

    def project(
        self,
        f: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Project function onto Hermite basis.

        Args:
            f: Function to project (accepts NumPy arrays).

        Returns:
            Array of coefficients [c_0, c_1, ..., c_n_max].
        """
        roots = self._quadrature.roots
        weights = self._quadrature.weights
        f_values = f(roots)

        coeffs = np.zeros(self.n_max + 1)
        for n, poly in enumerate(self._basis):
            h_values = poly.evaluate(roots)
            inner_product = np.sum(weights * f_values * h_values)
            coeffs[n] = inner_product / self._norms_sq[n]

        return coeffs

    def reconstruct(
        self,
        coeffs: np.ndarray,
        x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Reconstruct function from coefficients.

        Args:
            coeffs: Hermite coefficients [c_0, c_1, ..., c_n].
            x: Point(s) to evaluate at.

        Returns:
            Reconstructed function value(s).
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=np.float64)

        for n, c in enumerate(coeffs):
            if abs(c) > 1e-15:
                if n < len(self._basis):
                    result += c * self._basis[n].evaluate(x)
                else:
                    # Create polynomial on demand
                    poly = HermitePolynomial(n, "physicist")
                    result += c * poly.evaluate(x)

        return result

    def project_and_reconstruct(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Project function and reconstruct at given points.

        Args:
            f: Function to project.
            x: Points to evaluate reconstruction.

        Returns:
            Tuple of (coefficients, reconstructed_values).
        """
        coeffs = self.project(f)
        reconstructed = self.reconstruct(coeffs, x)
        return coeffs, reconstructed


def hermite_transform(
    f: Callable[[np.ndarray], np.ndarray],
    n_max: int,
    quadrature_points: Optional[int] = None,
    dps: int = 50
) -> Tuple[np.ndarray, GaussHermiteQuadrature]:
    """Compute Hermite transform (coefficients) of a function.

    Convenience function for single-shot projection.

    Args:
        f: Function to transform.
        n_max: Maximum degree.
        quadrature_points: Number of quadrature points.
        dps: Precision for computation.

    Returns:
        Tuple of (coefficients, quadrature object).
    """
    projector = HermiteProjection(n_max, quadrature_points, dps)
    return projector.project(f), projector._quadrature


def inverse_hermite_transform(
    coeffs: np.ndarray,
    x: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Reconstruct function from Hermite coefficients.

    Args:
        coeffs: Hermite coefficients.
        x: Points to evaluate at.

    Returns:
        Reconstructed values.
    """
    n_max = len(coeffs) - 1
    basis = hermite_numerical_basis(n_max, "physicist")

    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float64)

    for n, c in enumerate(coeffs):
        result += c * basis[n].evaluate(x)

    return result


def verify_quadrature_accuracy(n: int, dps: int = 50) -> dict:
    """Verify quadrature accuracy on known integrals.

    Tests against exact values:
        integral H_m * H_n * exp(-x^2) dx = delta_{mn} * 2^n * n! * sqrt(pi)

    Args:
        n: Number of quadrature points.
        dps: Precision for computation.

    Returns:
        Dictionary with verification results.
    """
    quad = GaussHermiteQuadrature(n, dps=dps)
    roots = quad.roots
    weights = quad.weights

    basis = hermite_numerical_basis(n - 1, "physicist")

    results = {
        "n_points": n,
        "orthogonality_errors": [],
        "norm_errors": []
    }

    for i in range(min(n, 20)):  # Test first 20 or n polynomials
        hi_vals = basis[i].evaluate(roots)

        # Check norm
        computed_norm_sq = np.sum(weights * hi_vals ** 2)
        expected_norm_sq = 2**i * math.factorial(i) * np.sqrt(np.pi)
        norm_error = abs(computed_norm_sq - expected_norm_sq) / expected_norm_sq
        results["norm_errors"].append(float(norm_error))

        # Check orthogonality with higher degree polynomials
        for j in range(i + 1, min(n, i + 5)):
            hj_vals = basis[j].evaluate(roots)
            inner_product = np.sum(weights * hi_vals * hj_vals)
            results["orthogonality_errors"].append(float(abs(inner_product)))

    results["max_norm_error"] = float(max(results["norm_errors"])) if results["norm_errors"] else 0.0
    results["max_ortho_error"] = float(max(results["orthogonality_errors"])) if results["orthogonality_errors"] else 0.0

    return results
