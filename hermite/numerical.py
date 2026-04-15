"""Hermite Polynomial Numerical Module - The Engine.

This module provides fast numerical evaluation of Hermite polynomials using NumPy/SciPy.
It is optimized for bulk data processing and array broadcasting operations.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple
import numpy as np
from numpy.polynomial import polynomial as P
try:
    from scipy.special import hermite as scipy_hermite_func
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class HermitePolynomial:
    """Fast numerical Hermite polynomial using NumPy."""
    
    def __init__(
        self,
        degree: int = 0,
        convention: str = "physicist",
        coeffs: Optional[Union[List[float], np.ndarray]] = None
    ):
        if degree < 0:
            raise ValueError(f"Degree must be non-negative, got {degree}")
        if convention not in ("physicist", "probabilist"):
            raise ValueError(f"Convention must be 'physicist' or 'probabilist'")
        
        self.degree = degree
        self.convention = convention
        
        if coeffs is not None:
            self._coeffs = np.asarray(coeffs, dtype=np.float64)
            self.degree = len(self._coeffs) - 1
        else:
            self._coeffs = self._generate_via_recurrence()
    
    def _generate_via_recurrence(self) -> np.ndarray:
        """Generate coefficients using Three-Term Recurrence.
        
        Returns coefficients in descending order [a_n, a_{n-1}, ..., a_0].
        H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x
        """
        if self.degree == 0:
            return np.array([1.0])
        
        # INTERNAL: ASCENDING order (index i = coeff of x^i)
        h_prev = np.array([1.0])  # H_0 = 1
        if self.convention == "physicist":
            h_curr = np.array([0.0, 2.0])  # H_1 = 2x
        else:
            h_curr = np.array([0.0, 1.0])  # He_1 = x
        
        if self.degree == 1:
            return h_curr[::-1]
        
        for k in range(1, self.degree):
            # Multiply by x: prepend zero (shift to higher powers)
            x_h_curr = np.concatenate([[0.0], h_curr])
            
            if self.convention == "physicist":
                term1 = 2.0 * x_h_curr
                # Pad at END (higher powers don't exist)
                padding_needed = len(term1) - len(h_prev)
                padded_prev = np.concatenate([h_prev, np.zeros(padding_needed)])
                term2 = 2.0 * k * padded_prev
            else:
                term1 = x_h_curr
                padding_needed = len(term1) - len(h_prev)
                padded_prev = np.concatenate([h_prev, np.zeros(padding_needed)])
                term2 = k * padded_prev
            
            h_next = term1 - term2
            
            # Remove trailing zeros (highest powers that are zero)
            nonzero_mask = np.abs(h_next) > 1e-15
            if not np.any(nonzero_mask):
                h_next = np.array([0.0])
            else:
                last_nonzero = len(h_next) - 1 - int(np.flipud(nonzero_mask).argmax())
                h_next = h_next[:last_nonzero + 1]
            
            h_prev, h_curr = h_curr, h_next
        
        # Convert from ascending to descending order
        return h_curr[::-1]
    
    def __repr__(self) -> str:
        conv = "H" if self.convention == "physicist" else "He"
        return f"HermitePolynomial({conv}_{self.degree}, convention=\'{self.convention}\')"
    
    def get_coefficients(self, ascending: bool = False) -> np.ndarray:
        if ascending:
            return self._coeffs[::-1].copy()
        return self._coeffs.copy()
    
    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        return np.polyval(self._coeffs, x)
    
    def derivative(self, order: int = 1) -> "HermitePolynomial":
        if order < 0:
            raise ValueError("Derivative order must be non-negative")
        if order == 0:
            return HermitePolynomial.from_coefficients(self._coeffs, self.convention)
        
        deriv_coeffs = np.polyder(self._coeffs, m=order)
        return HermitePolynomial.from_coefficients(deriv_coeffs, self.convention)
    
    def integrate(self) -> "HermitePolynomial":
        int_coeffs = np.polyint(self._coeffs, m=1, k=0)
        return HermitePolynomial.from_coefficients(int_coeffs, self.convention)
    
    @classmethod
    def from_coefficients(
        cls,
        coefficients: Union[List[float], np.ndarray],
        convention: str = "physicist"
    ) -> "HermitePolynomial":
        return cls(degree=0, convention=convention, coeffs=coefficients)
    
    @classmethod
    def from_symbolic(cls, symbolic_poly) -> "HermitePolynomial":
        coeffs = symbolic_poly.to_numerical_coeffs()
        return cls.from_coefficients(coeffs, symbolic_poly.convention)
    
    @classmethod
    def from_scipy_physicist(cls, n: int) -> "HermitePolynomial":
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for this method")
        h = scipy_hermite_func(n)
        coeffs = h.coeffs.tolist()
        return cls.from_coefficients(coeffs, "physicist")
    
    def verify_against_scipy(self, test_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for verification")
        
        if test_points is None:
            test_points = np.random.uniform(-3, 3, 10)
        
        our_values = self.evaluate(test_points)
        scipy_h = scipy_hermite_func(self.degree)
        scipy_values = scipy_h(test_points)
        
        nonzero_mask = np.abs(scipy_values) > 1e-10
        if np.any(nonzero_mask):
            rel_errors = np.abs(our_values[nonzero_mask] - scipy_values[nonzero_mask]) / np.abs(scipy_values[nonzero_mask])
            max_rel_error = float(np.max(rel_errors))
        else:
            max_rel_error = float(np.max(np.abs(our_values - scipy_values)))
        
        return test_points, our_values, max_rel_error


def hermite_numerical_basis(n_max: int, convention: str = "physicist") -> List[HermitePolynomial]:
    return [HermitePolynomial(n, convention) for n in range(n_max + 1)]