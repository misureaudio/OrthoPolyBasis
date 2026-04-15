# Numpy integration for Legendre polynomials
from typing import List, Optional
import warnings
import numpy as np

class NumpyLegendreGenerator:
    """Generate Legendre polynomials using NumPy for numerical computation.
    
    COEFFICIENT ORDER CONVENTION:
    This module uses DESCENDING order (highest degree first) by default,
    matching the core module and traditional mathematical convention:
        P_n(x) = c_0*x^n + c_1*x^(n-1) + ... + c_n
    
    Use get_coefficients_ascending() for NumPy's ascending order convention.
    """
    
    def __init__(self):
        try:
            from numpy.polynomial.legendre import legval, Legendre, leg2poly
            self.legval = legval
            self.Legendre = Legendre
            self.leg2poly = leg2poly
            self._numpy_available = True
        except ImportError:
            raise ImportError("NumPy is required for NumpyLegendreGenerator")
    
    def get_coefficients_descending(self, n: int) -> np.ndarray:
        """Get coefficients in DESCENDING order (highest degree first).
        
        This is the DEFAULT convention matching core.py and traditional math.
        P_n(x) = c_0*x^n + c_1*x^(n-1) + ... + c_n
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            Numpy array [c_0, c_1, ..., c_n] where P_n(x) = sum(c_i * x^(n-i))
            
        Example:
            >>> gen.get_coefficients_descending(2)
            array([ 1.5,  0. , -0.5])  # Represents 1.5*x^2 - 0.5
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        
        # Create Legendre basis coefficients: only nth term has coefficient 1
        leg_coeffs = np.zeros(n + 1)
        leg_coeffs[n] = 1.0
        
        # Convert from Legendre basis to standard polynomial form (ascending order)
        std_coeffs = self.leg2poly(leg_coeffs)
        
        # Reverse to descending order
        return std_coeffs[::-1]
    
    def get_coefficients_ascending(self, n: int) -> np.ndarray:
        """Get coefficients in ASCENDING order (constant term first).
        
        This matches NumPy's convention for Polynomial classes.
        P_n(x) = c_0 + c_1*x + ... + c_n*x^n
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            Numpy array [c_0, c_1, ..., c_n] where P_n(x) = sum(c_i * x^i)
            
        Example:
            >>> gen.get_coefficients_ascending(2)
            array([-0.5,  0. ,  1.5])  # Represents -0.5 + 1.5*x^2
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        
        # Create Legendre basis coefficients: only nth term has coefficient 1
        leg_coeffs = np.zeros(n + 1)
        leg_coeffs[n] = 1.0
        
        # Convert from Legendre basis to standard polynomial form (ascending order)
        return self.leg2poly(leg_coeffs)
    
    def get_coefficients(self, n: int) -> np.ndarray:
        """Get coefficients in DESCENDING order (default convention).
        
        Alias for get_coefficients_descending().
        
        DEPRECATED: Use get_coefficients_descending() or get_coefficients_ascending()
        explicitly to avoid confusion about coefficient ordering.
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            Numpy array in descending order [c_0, c_1, ..., c_n]
        """
        warnings.warn(
            "get_coefficients() is deprecated. Use get_coefficients_descending() "
            "or get_coefficients_ascending() explicitly to avoid confusion about ordering.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_coefficients_descending(n)
    
    def evaluate(self, x, n: int) -> float:
        """Evaluate P_n(x) at point(s) x.
        
        Uses NumPy's legval for direct evaluation in Legendre basis,
        which is more numerically stable than converting to monomial form.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        
        # Use legval for direct evaluation in Legendre basis (more stable)
        leg_coeffs = np.zeros(n + 1)
        leg_coeffs[n] = 1.0
        return self.legval(x, leg_coeffs)
    
    def evaluate_batch(self, xs: np.ndarray, n: int) -> np.ndarray:
        """Evaluate P_n(x) at multiple points.
        
        Uses NumPy's legval for direct evaluation in Legendre basis,
        which is more numerically stable than converting to monomial form.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        
        # Use legval for direct evaluation in Legendre basis (more stable)
        leg_coeffs = np.zeros(n + 1)
        leg_coeffs[n] = 1.0
        return self.legval(xs, leg_coeffs)
    
    def generate_basis(self, max_n: int) -> List[np.ndarray]:
        """Generate basis from P_0 to P_max_n as numpy arrays.
        
        Returns coefficients in DESCENDING order (default convention).
        Alias for generate_basis_descending().
        
        DEPRECATED: Use generate_basis_descending() or generate_basis_ascending()
        explicitly to avoid confusion about coefficient ordering.
        """
        warnings.warn(
            "generate_basis() is deprecated. Use generate_basis_descending() "
            "or generate_basis_ascending() explicitly to avoid confusion about ordering.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.generate_basis_descending(max_n)
    
    def generate_basis_descending(self, max_n: int) -> List[np.ndarray]:
        """Generate basis with coefficients in descending order."""
        if not isinstance(max_n, int) or max_n < 0:
            raise ValueError(f"max_n must be a non-negative integer, got {max_n}")
        return [self.get_coefficients_descending(n) for n in range(max_n + 1)]
    
    def generate_basis_ascending(self, max_n: int) -> List[np.ndarray]:
        """Generate basis with coefficients in ascending order."""
        if not isinstance(max_n, int) or max_n < 0:
            raise ValueError(f"max_n must be a non-negative integer, got {max_n}")
        return [self.get_coefficients_ascending(n) for n in range(max_n + 1)]


def generate_numpy_legendre(n: int) -> np.ndarray:
    """Convenience function to get the nth Legendre polynomial coefficients.
    
    Returns coefficients in DESCENDING order (default convention).
    Alias for generating via NumpyLegendreGenerator.get_coefficients_descending().
    
    DEPRECATED: Use generate_numpy_legendre_descending() or 
    generate_numpy_legendre_ascending() explicitly.
    """
    warnings.warn(
        "generate_numpy_legendre() is deprecated. Use generate_numpy_legendre_descending() "
        "or generate_numpy_legendre_ascending() explicitly to avoid confusion about ordering.",
        DeprecationWarning,
        stacklevel=2
    )
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")
    gen = NumpyLegendreGenerator()
    return gen.get_coefficients_descending(n)


def generate_numpy_legendre_descending(n: int) -> np.ndarray:
    """Get coefficients in descending order (default convention)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")
    gen = NumpyLegendreGenerator()
    return gen.get_coefficients_descending(n)


def generate_numpy_legendre_ascending(n: int) -> np.ndarray:
    """Get coefficients in ascending order (NumPy convention)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")
    gen = NumpyLegendreGenerator()
    return gen.get_coefficients_ascending(n)


def get_numpy_legendre_basis(max_n: int) -> List[np.ndarray]:
    """Convenience function to generate a basis of Legendre polynomials.
    
    Returns list of numpy arrays with coefficients in DESCENDING order.
    
    DEPRECATED: Use get_numpy_legendre_basis_descending() or 
    get_numpy_legendre_basis_ascending() explicitly.
    """
    warnings.warn(
        "get_numpy_legendre_basis() is deprecated. Use get_numpy_legendre_basis_descending() "
        "or get_numpy_legendre_basis_ascending() explicitly to avoid confusion about ordering.",
        DeprecationWarning,
        stacklevel=2
    )
    if not isinstance(max_n, int) or max_n < 0:
        raise ValueError(f"max_n must be a non-negative integer, got {max_n}")
    gen = NumpyLegendreGenerator()
    return gen.generate_basis_descending(max_n)


def get_numpy_legendre_basis_descending(max_n: int) -> List[np.ndarray]:
    """Generate basis with coefficients in descending order."""
    if not isinstance(max_n, int) or max_n < 0:
        raise ValueError(f"max_n must be a non-negative integer, got {max_n}")
    gen = NumpyLegendreGenerator()
    return gen.generate_basis_descending(max_n)


def get_numpy_legendre_basis_ascending(max_n: int) -> List[np.ndarray]:
    """Generate basis with coefficients in ascending order."""
    if not isinstance(max_n, int) or max_n < 0:
        raise ValueError(f"max_n must be a non-negative integer, got {max_n}")
    gen = NumpyLegendreGenerator()
    return gen.generate_basis_ascending(max_n)


def evaluate_numpy_legendre(x, n: int) -> float:
    """Convenience function to evaluate P_n(x)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")
    gen = NumpyLegendreGenerator()
    return gen.evaluate(x, n)
