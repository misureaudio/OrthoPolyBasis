"""
High-Precision Gauss-Legendre Quadrature
=========================================
Uses mpmath for arbitrary precision quadrature computation.
Essential for n > 100 or when extreme accuracy is required.
"""

from typing import List, Tuple, Optional
try:
    from mpmath import mp, mpf, mpi
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False


def _validate_n(n: int) -> None:
    """Validate that n is a positive integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")


class HighPrecisionGaussLegendre:
    """
    Compute Gauss-Legendre quadrature with arbitrary precision.
    
    Uses mpmath for all computations. Recommended when:
    - n > 100 (standard floats lose precision)
    - Integration requires more than ~15 decimal digits
    - Working near singularities or steep gradients
    """
    
    def __init__(self, dps: int = 50):
        """
        Initialize with desired precision.
        
        Args:
            dps: Decimal places of precision (default: 50)
        """
        if not MPMATH_AVAILABLE:
            raise ImportError("mpmath is required for HighPrecisionGaussLegendre")
        self.dps = dps
    
    def _evaluate_legendre(self, x: mpf, n: int) -> mpf:
        """Evaluate P_n(x) using three-term recurrence."""
        if n == 0:
            return mp.mpf("1")
        if n == 1:
            return x
        
        p_prev2 = mp.mpf("1")
        p_prev1 = x
        for k in range(2, n + 1):
            p_curr = ((2*k - 1) * x * p_prev1 - (k - 1) * p_prev2) / k
            p_prev2, p_prev1 = p_prev1, p_curr
        return p_curr
    
    def _evaluate_legendre_derivative(self, x: mpf, n: int) -> mpf:
        """Evaluate P_n'(x) using the identity."""
        if n == 0:
            return mp.mpf("0")
        if n == 1:
            return mp.mpf("1")
        
        pn = self._evaluate_legendre(x, n)
        pn_1 = self._evaluate_legendre(x, n - 1)
        
        x2_minus_1 = x * x - mp.mpf("1")
        if abs(x2_minus_1) < mp.mpf("1e-30"):
            # Near endpoints
            sign = mp.mpf("1") if x > 0 else mp.power(mp.mpf("-1"), n - 1)
            return sign * n * (n + 1) / mp.mpf("2")
        
        return n * (x * pn - pn_1) / x2_minus_1
    
    def compute(self, n: int, method: str = "newton_raphson") -> Tuple[List[mpf], List[mpf]]:
        """
        Compute high-precision Gauss-Legendre quadrature.
        
        Args:
            n: Number of quadrature points
            method: Currently only "newton_raphson" supported in high precision
                   (Golub-Welsch would require mpmath eigenvalue solver)
            
        Returns:
            (nodes, weights) as lists of mp.mpf objects
        """
        _validate_n(n)
        
        old_dps = mp.dps
        mp.dps = self.dps
        
        try:
            nodes = []
            weights = []
            
            for i in range(n):
                # Initial guess: cosine distribution
                theta = mp.pi * (4 * (i + 1) - 1) / (4 * n + 2)
                x = mp.cos(theta)
                
                # Newton-Raphson iteration
                for _ in range(100):  # More iterations for high precision
                    pn = self._evaluate_legendre(x, n)
                    dpn = self._evaluate_legendre_derivative(x, n)
                    
                    if abs(dpn) < mp.mpf("1e-50"):
                        break
                        
                    dx = -pn / dpn
                    x += dx
                    
                    if abs(dx) < mp.power(10, -self.dps + 5):
                        break
                
                nodes.append(x)
                
                # Compute weight
                _, dpn = self._evaluate_legendre_derivative(x, n), self._evaluate_legendre_derivative(x, n)
                dpn = self._evaluate_legendre_derivative(x, n)
                w = mp.mpf("2") / ((mp.mpf("1") - x * x) * dpn * dpn)
                weights.append(w)
            
            # Sort by node value
            combined = sorted(zip(nodes, weights))
            nodes = [c[0] for c in combined]
            weights = [c[1] for c in combined]
            
            return nodes, weights
        finally:
            mp.dps = old_dps
    
    def integrate(self, f, n: int) -> mpf:
        """
        Integrate function f over [-1, 1] with n quadrature points.
        
        Args:
            f: Function to integrate (should accept mp.mpf and return mp.mpf)
            n: Number of quadrature points
            
        Returns:
            High-precision integral approximation
        """
        nodes, weights = self.compute(n)
        result = mp.mpf("0")
        for node, weight in zip(nodes, weights):
            result += weight * f(node)
        return result


# Convenience functions
def gauss_legendre_high_precision(n: int, dps: int = 50) -> Tuple[List[mpf], List[mpf]]:
    """
    Compute Gauss-Legendre quadrature with arbitrary precision.
    
    Args:
        n: Number of quadrature points
        dps: Decimal places of precision (default: 50)
        
    Returns:
        (nodes, weights) as lists of mp.mpf objects
    """
    quad = HighPrecisionGaussLegendre(dps=dps)
    return quad.compute(n)
