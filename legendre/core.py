# Core Legendre polynomial generator
from typing import List, Union
import warnings

def _validate_n(n: int) -> None:
    """Validate that n is a non-negative integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

class LegendreGenerator:
    """Generate Legendre polynomials using the three-term recurrence relation.
    
    COEFFICIENT ORDER CONVENTION:
    This module uses DESCENDING order (highest degree first) by default,
    matching traditional mathematical convention:
        P_n(x) = c_0*x^n + c_1*x^(n-1) + ... + c_n
    
    Use get_coefficients_ascending() for ascending order convention.
    """
    def __init__(self):
        self._cache_desc = {}
        self._cache_asc = {}

    def get_coefficients_descending(self, n: int) -> List[float]:
        """Get coefficients in DESCENDING order (highest degree first).
        
        This is the DEFAULT convention matching traditional math.
        P_n(x) = c_0*x^n + c_1*x^(n-1) + ... + c_n
        
        WARNING: For n > 60, these float coefficients may lose precision due to
        catastrophic cancellation. The coefficients grow very large (e.g., for n=50,
        coefficients exceed 10^14), leading to numerical instability.
        
        DO NOT use these coefficients for evaluation at high n. Use evaluate() instead,
        which uses a stable three-term recurrence algorithm.
        
        For high-precision coefficient computation, use sympy_integration or
        mpmath_integration modules.
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            List of coefficients [c_0, c_1, ..., c_n] where P_n(x) = sum(c_i * x^(n-i))
            
        Example:
            >>> gen.get_coefficients_descending(2)
            [1.5, 0.0, -0.5]  # Represents 1.5*x^2 - 0.5
        """
        _validate_n(n)
        if n in self._cache_desc:
            return self._cache_desc[n].copy()
        
        if n == 0:
            return [1.0]
        if n == 1:
            return [1.0, 0.0]

        # Initialize P0 and P1 in ascending order (index = power)
        # This makes the math much easier to reason about
        p_prev2 = [1.0]
        p_prev1 = [0.0, 1.0]
        
        for k in range(2, n + 1):
            # (k)P_k = (2k-1)xP_{k-1} - (k-1)P_{k-2}
            # new_p degree is k
            current = [0.0] * (k + 1)
            
            # (2k-1)x * P_{k-1}
            # Shifting P_{k-1} by 1 index is equivalent to multiplying by x
            factor1 = (2 * k - 1)
            for i, coeff in enumerate(p_prev1):
                current[i + 1] = factor1 * coeff
                
            # -(k-1) * P_{k-2}
            factor2 = (k - 1)
            for i, coeff in enumerate(p_prev2):
                current[i] -= factor2 * coeff
                
            # Divide all by k
            for i in range(len(current)):
                current[i] /= k
                
            p_prev2 = p_prev1
            p_prev1 = current

        # Convert to descending order
        coeffs = p_prev1[::-1]
        self._cache_desc[n] = coeffs
        return coeffs.copy()
    
    def get_coefficients_ascending(self, n: int) -> List[float]:
        """Get coefficients in ASCENDING order (constant term first).
        
        P_n(x) = c_0 + c_1*x + ... + c_n*x^n
        
        WARNING: For n > 60, these float coefficients may lose precision due to
        catastrophic cancellation. See get_coefficients_descending() for details.
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            List of coefficients [c_0, c_1, ..., c_n] where P_n(x) = sum(c_i * x^i)
            
        Example:
            >>> gen.get_coefficients_ascending(2)
            [-0.5, 0.0, 1.5]  # Represents -0.5 + 1.5*x^2
        """
        _validate_n(n)
        if n in self._cache_asc:
            return self._cache_asc[n].copy()
        
        # Get descending and reverse
        desc_coeffs = self.get_coefficients_descending(n)
        asc_coeffs = desc_coeffs[::-1]
        
        self._cache_asc[n] = asc_coeffs
        return asc_coeffs.copy()
    
    def get_coefficients(self, n: int) -> List[float]:
        """Get coefficients in DESCENDING order (default convention).
        
        Alias for get_coefficients_descending().
        
        DEPRECATED: Use get_coefficients_descending() or 
        get_coefficients_ascending() explicitly to avoid confusion about ordering.
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            List of coefficients in descending order
        """
        warnings.warn(
            "get_coefficients() is deprecated. Use get_coefficients_descending() "
            "or get_coefficients_ascending() explicitly to avoid confusion about ordering.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_coefficients_descending(n)

    def _evaluate_stable(self, x: float, n: int) -> float:
        """
        Evaluate P_n(x) using forward three-term recurrence.
        Numerically stable O(n) algorithm that avoids monomial powers.
        Uses: (k+1)*P_{k+1}(x) = (2k-1)*x*P_k(x) - (k-1)*P_{k-1}(x)
        """
        if n == 0:
            return 1.0
        p_prev, p_curr = 1.0, float(x)  # P_0, P_1
        for k in range(2, n + 1):
            p_next = ((2*k - 1) * x * p_curr - (k - 1) * p_prev) / k
            p_prev, p_curr = p_curr, p_next
        return p_curr
    
    def evaluate(self, x: float, n: int) -> float:
        _validate_n(n)
        # Use stable forward recurrence to avoid monomial instability
        return self._evaluate_stable(x, n)
    
    def evaluate_batch(self, xs: List[float], n: int) -> List[float]:
        _validate_n(n)
        return [self.evaluate(x, n) for x in xs]
    
    def derivative_coefficients(self, n: int) -> List[float]:
        """Get coefficients of the derivative of the nth Legendre polynomial.
        
        WARNING: This method computes derivatives by manipulating monomial coefficients,
        which is numerically unstable for high n (typically n > 60). The instability
        arises from catastrophic cancellation when working with large coefficients.
        
        For stable derivative computation at high degrees, consider:
        1. Using evaluate() on the identity: P_n'(x) = n/(x²-1) * (x*P_n(x) - P_{n-1}(x))
        2. Using the sympy_integration module for symbolic derivatives
        3. Using stable_operations.py which works directly in Legendre basis
        
        Args:
            n: Degree of Legendre polynomial
            
        Returns:
            Coefficients of d/dx(P_n(x)) in descending order
        """
        _validate_n(n)
        if n == 0:
            return [0.0]
        coeffs = self.get_coefficients_descending(n)
        # Derivative: d/dx(c*x^k) = c*k*x^(k-1)
        degree = len(coeffs) - 1
        deriv_coeffs = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power > 0:
                deriv_coeffs.append(c * power)
        return deriv_coeffs
    
    def integral_coefficients(self, n: int) -> List[float]:
        """Get coefficients of the indefinite integral of the nth Legendre polynomial.
        
        WARNING: This method computes integrals by manipulating monomial coefficients,
        which is numerically unstable for high n (typically n > 60).
        
        For stable integral computation at high degrees, consider:
        1. Using the identity: ?P_n(x)dx = (P_{n+1}(x) - P_{n-1}(x))/(2n+1) for n >= 1
        2. Using the sympy_integration module for symbolic integrals
        3. Using stable_operations.py which works directly in Legendre basis
        
        Args:
            n: Degree of Legendre polynomial
            
        Returns:
            Coefficients of ?P_n(x)dx in descending order (constant term = 0)
        """
        _validate_n(n)
        coeffs = self.get_coefficients_descending(n)
        # Integral: ?(c*x^k)dx = c/(k+1)*x^(k+1)
        degree = len(coeffs) - 1
        int_coeffs = []
        for i, c in enumerate(coeffs):
            power = degree - i
            int_coeffs.append(c / (power + 1))
        # Add constant of integration (0)
        int_coeffs.append(0.0)
        return int_coeffs
    
    def generate_basis(self, max_n: int) -> List[List[float]]:
        """Generate basis coefficients from P_0 to P_max_n.
        
        Returns coefficients in DESCENDING order (default convention).
        
        DEPRECATED: Use generate_basis_descending() or 
        generate_basis_ascending() explicitly to avoid confusion about ordering.
        """
        warnings.warn(
            "generate_basis() is deprecated. Use generate_basis_descending() "
            "or generate_basis_ascending() explicitly to avoid confusion about ordering.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.generate_basis_descending(max_n)
    
    def generate_basis_descending(self, max_n: int) -> List[List[float]]:
        """Generate basis coefficients in descending order."""
        _validate_n(max_n)
        return [self.get_coefficients_descending(n) for n in range(max_n + 1)]
    
    def generate_basis_ascending(self, max_n: int) -> List[List[float]]:
        """Generate basis coefficients in ascending order."""
        _validate_n(max_n)
        return [self.get_coefficients_ascending(n) for n in range(max_n + 1)]

# Convenience functions
def legendre_polynomial(n: int, x: float) -> float:
    """Convenience function to evaluate P_n(x)."""
    _validate_n(n)
    return LegendreGenerator().evaluate(x, n)

def legendre_coefficients(n: int) -> List[float]:
    """Get coefficients in descending order.
    
    DEPRECATED: Use legendre_coefficients_descending() or 
    legendre_coefficients_ascending() explicitly.
    """
    warnings.warn(
        "legendre_coefficients() is deprecated. Use legendre_coefficients_descending() "
        "or legendre_coefficients_ascending() explicitly to avoid confusion about ordering.",
        DeprecationWarning,
        stacklevel=2
    )
    _validate_n(n)
    return LegendreGenerator().get_coefficients_descending(n)

def legendre_coefficients_descending(n: int) -> List[float]:
    """Get coefficients in descending order (default convention)."""
    _validate_n(n)
    return LegendreGenerator().get_coefficients_descending(n)

def legendre_coefficients_ascending(n: int) -> List[float]:
    """Get coefficients in ascending order."""
    _validate_n(n)
    return LegendreGenerator().get_coefficients_ascending(n)

legendre_derivative = lambda n: (_validate_n(n), LegendreGenerator().derivative_coefficients(n))[1]
legendre_integral = lambda n: (_validate_n(n), LegendreGenerator().integral_coefficients(n))[1]
