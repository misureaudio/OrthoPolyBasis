# Numpy integration for Chebyshev polynomials
from typing import List, Optional
import numpy as np

class NumpyChebyshevGenerator:
    """Generate Chebyshev polynomials using NumPy for numerical computation."""
    
    def __init__(self):
        try:
            from numpy.polynomial.chebyshev import chebval, Chebyshev
            self.chebval = chebval
            self.Chebyshev = Chebyshev
            self._numpy_available = True
        except ImportError:
            raise ImportError("NumPy is required for NumpyChebyshevGenerator")
    
    def get_coefficients(self, n: int) -> np.ndarray:
        """Get coefficients of the nth Chebyshev polynomial as numpy array.
        Returns coefficients in ascending order (constant term first).
        T_n(x) = c_0 + c_1*x + ... + c_n*x^n
        """
        # Use numpy's built-in chebyshev polynomial generation
        poly = self.Chebyshev.basis(n)
        return np.array(poly.coef)
    
    def get_coefficients_descending(self, n: int) -> np.ndarray:
        """Get coefficients in descending order (highest degree first)."""
        coeffs = self.get_coefficients(n)
        return coeffs[::-1]
    
    def evaluate(self, x, n: int) -> float:
        """Evaluate T_n(x) at point(s) x."""
        coeffs = self.get_coefficients(n)
        return self.chebval(x, coeffs)
    
    def evaluate_batch(self, xs: np.ndarray, n: int) -> np.ndarray:
        """Evaluate T_n(x) at multiple points."""
        coeffs = self.get_coefficients(n)
        return self.chebval(xs, coeffs)
    
    def generate_basis(self, max_n: int) -> List[np.ndarray]:
        """Generate basis from T_0 to T_max_n as numpy arrays.
        Each array contains coefficients in ascending order.
        """
        return [self.get_coefficients(n) for n in range(max_n + 1)]
    
    def generate_basis_descending(self, max_n: int) -> List[np.ndarray]:
        """Generate basis with coefficients in descending order."""
        return [self.get_coefficients_descending(n) for n in range(max_n + 1)]


def generate_numpy_chebyshev(n: int) -> np.ndarray:
    """Convenience function to get the nth Chebyshev polynomial coefficients.
    Returns coefficients in ascending order (constant term first).
    """
    gen = NumpyChebyshevGenerator()
    return gen.get_coefficients(n)


def get_numpy_chebyshev_basis(max_n: int) -> List[np.ndarray]:
    """Convenience function to generate a basis of Chebyshev polynomials.
    Returns list of numpy arrays with coefficients in ascending order.
    """
    gen = NumpyChebyshevGenerator()
    return gen.generate_basis(max_n)


def evaluate_numpy_chebyshev(x, n: int) -> float:
    """Convenience function to evaluate T_n(x)."""
    gen = NumpyChebyshevGenerator()
    return gen.evaluate(x, n)