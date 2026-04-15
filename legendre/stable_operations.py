"""
Stable Legendre-Basis Operations
=================================
Implements derivative and integral operations that remain in the Legendre basis,
avoiding conversion to monomial basis which is numerically unstable for high degrees.

Key recurrence relations:
-------------------------
1. Derivative (expands P_n' in Legendre basis):
   P_n'(x) = sum of odd/even P_k(x) depending on parity of n
   
   Specifically: P_n'(x) = (2m+1) * sum_{m=0}^{floor((n-1)/2)} P_{n-1-2m}(x)
   
   Or equivalently using:
   (2n+1)*P_n(x) = P_{n+1}'(x) - P_{n-1}'(x)

2. Integral (expands integral in Legendre basis):
   int(P_n)(x) = (P_{n+1}(x) - P_{n-1}(x)) / (2n + 1)  for n >= 1
   int(P_0)(x) = x = P_1(x)

3. Multiplication by x:
   x*P_n(x) = n/(2n+1)*P_{n-1}(x) + (n+1)/(2n+1)*P_{n+1}(x)
"""

from typing import List
import numpy as np

def _validate_non_negative(n: int) -> None:
    """Validate n is non-negative integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")


class LegendreBasisOperations:
    """
    Operations that work directly in the Legendre basis.
    
    A polynomial expressed in Legendre basis:
        p(x) = sum_{k=0}^{n} c_k * P_k(x)
    
    is represented as coefficient array [c_0, c_1, ..., c_n] (ascending order).
    """
    
    @staticmethod
    def derivative_legendre_basis(coeffs: np.ndarray) -> np.ndarray:
        """
        Compute derivative of polynomial expressed in Legendre basis.
        
        Given p(x) = sum_{k=0}^{n} c_k * P_k(x),
        returns coefficients d_j such that p'(x) = sum_{j=0}^{n-1} d_j * P_j(x)
        
        Uses the formula:
            P_n'(x) = sum_{j odd/even} (2j+1) * P_j(x)
                   where j runs from (n-1) down to 0 or 1 with step 2
        
        Equivalently: P_n'(x) = (2n-1)P_{n-1}(x) + (2n-5)P_{n-3}(x) + ...
        
        Args:
            coeffs: Array of Legendre coefficients [c_0, c_1, ..., c_n]
                   representing sum c_k * P_k(x)
        
        Returns:
            Array of Legendre coefficients for the derivative
        """
        n = len(coeffs) - 1  # degree
        if n == 0:
            return np.array([0.0])  # derivative of constant is zero
        
        # Derivative reduces degree by 1
        deriv_coeffs = np.zeros(n)
        
        for k in range(1, n + 1):  # k is the original degree
            if coeffs[k] == 0:
                continue
            # P_k' contributes to P_{k-1}, P_{k-3}, P_{k-5}, ...
            # Coefficient of P_j in P_k' is (2j+1) where j has same parity as k-1
            for j in range(k - 1, -1, -2):
                deriv_coeffs[j] += coeffs[k] * (2*j + 1)
        
        return deriv_coeffs
    
    @staticmethod
    def integral_legendre_basis(coeffs: np.ndarray, C: float = 0.0) -> np.ndarray:
        """
        Compute indefinite integral of polynomial in Legendre basis.
        
        Given p(x) = sum_{k=0}^{n} c_k * P_k(x),
        returns coefficients I_j such that int(p)(x) = sum_{j=0}^{n+1} I_j * P_j(x) + C
        
        Uses the formula:
            int(P_n)(x) dx = (P_{n+1}(x) - P_{n-1}(x)) / (2n + 1)  for n >= 1
            int(P_0)(x) dx = x = P_1(x)
        
        Args:
            coeffs: Array of Legendre coefficients [c_0, c_1, ..., c_n]
            C: Constant of integration (added as coefficient of P_0)
        
        Returns:
            Array of Legendre coefficients for the integral
        """
        n = len(coeffs) - 1  # degree
        if n < 0:
            return np.array([C])
        
        # Integral increases degree by 1
        int_coeffs = np.zeros(n + 2)
        
        # Handle P_0 separately: int(P_0) = x = P_1
        int_coeffs[1] += coeffs[0]
        
        # For k >= 1: int(P_k) = (P_{k+1} - P_{k-1}) / (2k + 1)
        for k in range(1, n + 1):
            if coeffs[k] == 0:
                continue
            factor = coeffs[k] / (2*k + 1)
            int_coeffs[k + 1] += factor
            int_coeffs[k - 1] -= factor
        
        # Add constant of integration to P_0 coefficient
        int_coeffs[0] += C
        
        return int_coeffs
    
    @staticmethod
    def multiply_by_x_legendre_basis(coeffs: np.ndarray) -> np.ndarray:
        """
        Multiply polynomial in Legendre basis by x.
        
        Uses the three-term recurrence:
            x * P_n(x) = n/(2n+1) * P_{n-1}(x) + (n+1)/(2n+1) * P_{n+1}(x)
        
        Args:
            coeffs: Array of Legendre coefficients [c_0, c_1, ..., c_n]
        
        Returns:
            Array of Legendre coefficients for x * p(x), degree n+1
        """
        n = len(coeffs) - 1  # degree
        if n < 0:
            return np.array([0.0])
        
        result = np.zeros(n + 2)
        
        for k in range(n + 1):
            if coeffs[k] == 0:
                continue
            if k > 0:
                # Contribution to P_{k-1}
                result[k - 1] += coeffs[k] * k / (2*k + 1)
            # Contribution to P_{k+1}
            result[k + 1] += coeffs[k] * (k + 1) / (2*k + 1)
        
        return result
    
    @staticmethod
    def convert_from_monomial(monomial_coeffs: np.ndarray) -> np.ndarray:
        """
        Convert from monomial basis to Legendre basis.
        
        Given coefficients a_k for sum a_k * x^k,
        returns Legendre coefficients c_j for sum c_j * P_j(x).
        
        Uses the fact that x^n can be expressed as combination of P_n, P_{n-2}, ...
        """
        n = len(monomial_coeffs) - 1
        if n < 0:
            return np.array([0.0])
        
        legendre_coeffs = np.zeros(n + 1)
        
        # Build up x^k in Legendre basis iteratively
        # Start with x^0 = 1 = P_0
        x_power_in_legendre = np.array([1.0])  # Represents x^0 = P_0
        legendre_coeffs[0] += monomial_coeffs[0]
        
        for k in range(1, n + 1):
            # Multiply by x to get x^k from x^{k-1}
            x_power_in_legendre = LegendreBasisOperations.multiply_by_x_legendre_basis(x_power_in_legendre)
            legendre_coeffs[:len(x_power_in_legendre)] += monomial_coeffs[k] * x_power_in_legendre
        
        return legendre_coeffs
    
    @staticmethod
    def convert_to_monomial(legendre_coeffs: np.ndarray) -> np.ndarray:
        """
        Convert from Legendre basis to monomial basis.
        
        This is less stable than staying in Legendre basis, but provided
        for compatibility with code expecting monomial coefficients.
        """
        n = len(legendre_coeffs) - 1
        if n < 0:
            return np.array([0.0])
        
        # Start with zero polynomial in monomial basis
        result = np.zeros(n + 1)
        
        # Add each Legendre term converted to monomial
        from numpy.polynomial.legendre import leg2poly
        for k in range(n + 1):
            if legendre_coeffs[k] == 0:
                continue
            # Create Legendre basis vector for P_k
            leg_basis = np.zeros(k + 1)
            leg_basis[k] = 1.0
            # Convert to monomial and scale
            mono = leg2poly(leg_basis)
            result[:len(mono)] += legendre_coeffs[k] * mono
        
        return result


def derivative_legendre(coeffs: np.ndarray) -> np.ndarray:
    """Convenience function for Legendre-basis derivative."""
    return LegendreBasisOperations.derivative_legendre_basis(np.asarray(coeffs))


def integral_legendre(coeffs: np.ndarray, C: float = 0.0) -> np.ndarray:
    """Convenience function for Legendre-basis integral."""
    return LegendreBasisOperations.integral_legendre_basis(np.asarray(coeffs), C)
