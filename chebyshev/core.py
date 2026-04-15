from typing import List

class ChebyshevGenerator:
    def __init__(self):
        self._cache = {}

    def get_monomial_coefficients(self, n: int) -> List[float]:
        """UNSTABLE for large n. Returns coeffs for 1, x, x^2..."""
        # ... (Insert the previous list-based recurrence logic here) ...
        pass

    def get_derivative_series(self, n: int) -> List[float]:
        """
        STABLE. Returns coefficients of the derivative in the Chebyshev basis.
        Example: T'_3 = 3 * (2*T_2 + T_0) -> returns [3.0, 0.0, 6.0]
        Index i corresponds to coefficient of T_i.
        """
        if n == 0:
            return [0.0]
        
        # Resulting coeffs for [T_0, T_1, ..., T_{n-1}]
        deriv_coeffs = [0.0] * n
        
        # Identity: T'_n = n * (2*T_{n-1} + 2*T_{n-3} + ...)
        # Last term is T_0 if n is odd, 2*T_1 if n is even
        for i in range(n - 1, -1, -2):
            if i == 0:
                deriv_coeffs[i] = float(n) # The T_0 term doesn't get a factor of 2
            else:
                deriv_coeffs[i] = float(2 * n)
        return deriv_coeffs

    def get_integral_series(self, n: int) -> List[float]:
        """
        STABLE. Returns coefficients of the integral in the Chebyshev basis.
        Index i corresponds to coefficient of T_i.
        """
        # Result will be up to T_{n+1}
        integ_coeffs = [0.0] * (n + 2)
        
        if n == 0:
            # ∫T_0 dx = T_1
            integ_coeffs[1] = 1.0
        elif n == 1:
            # ∫T_1 dx = 0.25*T_2 + 0.5*T_0 (plus constant, we use 0)
            integ_coeffs[0] = 0.5
            integ_coeffs[2] = 0.25
        else:
            # ∫T_n dx = 0.5 * [ T_{n+1}/(n+1) - T_{n-1}/(n-1) ]
            integ_coeffs[n + 1] = 1.0 / (2 * (n + 1))
            integ_coeffs[n - 1] = -1.0 / (2 * (n - 1))
            
        return integ_coeffs

    def evaluate_series(self, x: float, series_coeffs: List[float]) -> float:
        """
        Evaluates a Chebyshev series sum(a_i * T_i(x)) using Clenshaw's Algorithm.
        This is the most numerically stable way to evaluate any Chebyshev polynomial.
        """
        if not series_coeffs:
            return 0.0
        
        n = len(series_coeffs) - 1
        b_k = 0.0
        b_k1 = 0.0
        b_k2 = 0.0
        
        # Clenshaw's backward recurrence
        for k in range(n, 0, -1):
            b_k = series_coeffs[k] + 2 * x * b_k1 - b_k2
            b_k2 = b_k1
            b_k1 = b_k
            
        return series_coeffs[0] + x * b_k1 - b_k2

# --- Updated Convenience API ---

_GEN = ChebyshevGenerator()

def chebyshev_derivative_stable(n: int, x: float) -> float:
    """Calculates T'_n(x) without ever using monomials."""
    coeffs = _GEN.get_derivative_series(n)
    return _GEN.evaluate_series(x, coeffs)

def chebyshev_integral_stable(n: int, x: float) -> float:
    """Calculates integral of T_n(x) without ever using monomials."""
    coeffs = _GEN.get_integral_series(n)
    return _GEN.evaluate_series(x, coeffs)

# --- Legacy API Compatibility ---

def chebyshev_coefficients(n: int) -> List[float]:
    """Returns monomial coefficients for T_n(x)."""
    return _GEN.get_monomial_coefficients(n)

def chebyshev_derivative(n: int, x: float) -> float:
    """Calculates T'_n(x)."""
    return chebyshev_derivative_stable(n, x)

def chebyshev_integral(n: int, x: float) -> float:
    """Calculates integral of T_n(x)."""
    return chebyshev_integral_stable(n, x)