# Quadrature and Roots module for Chebyshev polynomials
from typing import List, Callable, Optional
import math

FFT_THRESHOLD = 64  # Use FFT method when n >= this value

class ChebyshevQuadrature:
    def __init__(self):
        self._cache = {}  # Cache for (type, n) -> (nodes, weights)
    def __init__(self):
        self._cache = {}  # Cache for (type, n) -> (nodes, weights)
    """Chebyshev quadrature and roots computation.
    
    Provides:
    - Zeros of T_n(x) (Gauss-Chebyshev nodes)
    - Extrema of T_n(x) (Clenshaw-Curtis nodes)
    - Gauss-Chebyshev quadrature for weighted integrals
    - Clenshaw-Curtis quadrature with automatic O(n2)/O(n log n) selection
    """
    
    def get_zeros(self, n: int) -> List[float]:
        """Compute the n zeros of T_n(x).
        
        These are the Gauss-Chebyshev quadrature nodes.
        Closed-form formula: x_k = cos((2k-1)*pi/(2n)) for k=1..n
        
        Args:
            n: Degree of Chebyshev polynomial (number of zeros)
            
        Returns:
            List of n zeros in ascending order
            
        Example:
            >>> q = ChebyshevQuadrature()
            >>> q.get_zeros(4)
            [-0.9238795, -0.3826834, 0.3826834, 0.9238795]
        """
        if n <= 0:
            return []
        
        zeros = []
        for k in range(1, n + 1):
            x_k = math.cos((2 * k - 1) * math.pi / (2 * n))
            zeros.append(x_k)
        return list(reversed(zeros))  # Return in ascending order
    
    def get_extrema_points(self, n: int) -> List[float]:
        """Compute the n+1 extrema points of T_n(x).
        
        These are the Clenshaw-Curtis quadrature nodes (where |T_n| = 1).
        Formula: x_k = cos(k*pi/n) for k=0..n
        
        Args:
            n: Degree of Chebyshev polynomial
            
        Returns:
            List of n+1 extrema points in descending order (from 1 to -1)
            
        Example:
            >>> q = ChebyshevQuadrature()
            >>> q.get_extrema_points(3)
            [1.0, 0.5, -0.5, -1.0]
        """
        if n < 0:
            return []
        
        points = []
        for k in range(n + 1):
            x_k = math.cos(k * math.pi / n) if n > 0 else 1.0
            points.append(x_k)
        return points
    
    def get_gauss_chebyshev_weights(self, n: int) -> List[float]:
        """Compute weights for Gauss-Chebyshev quadrature.
        
        All weights are equal to pi/n for Gauss-Chebyshev quadrature.
        
        Args:
            n: Number of quadrature points
            
        Returns:
            List of n weights (all equal to pi/n)
        """
        if n <= 0:
            return []
        return [math.pi / n] * n
    
    def gauss_chebyshev_quadrature(self, f: Callable[[float], float], n: int) -> float:
        """Compute Gauss-Chebyshev quadrature of f(x).
        
        Approximates the integral:
            ?_{-1}^{1} f(x) / sqrt(1-x2) dx 2 (p/n) * S_{k=1}^{n} f(x_k)
        
        where x_k are the zeros of T_n(x).
        
        Args:
            f: Function to integrate
            n: Number of quadrature points (higher = more accurate)
            
        Returns:
            Approximation of the weighted integral
            
        Example:
            >>> q = ChebyshevQuadrature()
            >>> result = q.gauss_chebyshev_quadrature(lambda x: x**2, n=64)
            # Computes ?_{-1}^{1} x2/v(1-x2) dx 2 p/2
        """
        if n <= 0:
            return 0.0
        
        zeros = self.get_zeros(n)
        weight = math.pi / n
        
        total = 0.0
        for x_k in zeros:
            total += f(x_k)
        
        return weight * total
    
    def clenshaw_curtis_quadrature(self, f: Callable[[float], float], n: int) -> float:
        """Compute Clenshaw-Curtis quadrature of f(x).
        
        Approximates the integral:
            ?_{-1}^{1} f(x) dx
        
        Uses extrema points (Chebyshev nodes of the second kind) as sample points.
        Automatically selects O(n2) method for small n, FFT-based O(n log n) for large n.
        
        Args:
            f: Function to integrate
            n: Degree (uses n+1 points including endpoints)
            
        Returns:
            Approximation of ?_{-1}^{1} f(x) dx
            
        Example:
            >>> q = ChebyshevQuadrature()
            >>> result = q.clenshaw_curtis_quadrature(lambda x: math.exp(x), n=32)
            # Computes ?_{-1}^{1} e^x dx = e - 1/e 2 2.3504
        """
        if n < 0:
            return 0.0
        
        # Cache lookup for nodes and weights
        cache_key = ("cc", n)
        if cache_key not in self._cache:
            points = self.get_extrema_points(n)
            weights = self._compute_clenshaw_curtis_weights(n)
            self._cache[cache_key] = (points, weights)
        
        points, weights = self._cache[cache_key]
        
        # Compute integral using cached nodes and weights
        total = 0.0
        for x_k, w_k in zip(points, weights):
            total += w_k * f(x_k)
        return total
    
    def _clenshaw_curtis_quadrature_direct(self, f: Callable[[float], float], n: int) -> float:
        """O(n2) Clenshaw-Curtis quadrature for small n."""
        if n == 0:
            return 2.0 * f(1.0)
        
        points = self.get_extrema_points(n)
        weights = self._compute_clenshaw_curtis_weights(n)
        
        total = 0.0
        for x_k, w_k in zip(points, weights):
            total += w_k * f(x_k)
        
        return total
    
    def _clenshaw_curtis_quadrature_fft(self, f, n):
        from clencurt import clencurt
        # from .clencurt import clencurt
        nodes, weights = clencurt(n)  # Get numpy arrays
        total = 0.0
        for x_k, w_k in zip(nodes, weights):  # Point-by-point!
            total += w_k * f(x_k)
        return total

    """
    def _clenshaw_curtis_quadrature_fft(self, f: Callable[[float], float], n: int) -> float:
        # O(n log n) Clenshaw-Curtis quadrature using FFT for large n.
        try:
            import numpy as np
            from clencurt import clencurt
            
            nodes, weights = clencurt(n)
            return float(np.sum(weights * f(nodes)))
        except ImportError:
            # Fallback to direct method if numpy/clencurt not available
            return self._clenshaw_curtis_quadrature_direct(f, n)
    """

    def _compute_clenshaw_curtis_weights(self, n: int) -> List[float]:
        """Compute Clenshaw-Curtis quadrature weights in O(n2).
        
        Uses the standard algorithm from numerical analysis literature.
        Reference: Trefethen, "Spectral Methods in MATLAB"
        """
        if n == 0:
            return [2.0]
        if n == 1:
            return [1.0, 1.0]
        
        weights = [0.0] * (n + 1)
        for k in range(n + 1):
            # c_k: 1 at endpoints, 2 in the interior
            ck = 1.0 if (k == 0 or k == n) else 2.0
            
            s = 1.0  # The j=0 term
            # Sum terms for j = 1 to floor(n/2)
            for j in range(1, (n // 2) + 1):
                val = 2.0 / (1.0 - 4.0 * j**2)
                # b_j: 0.5 if 2j == n, else 1.0
                bj = 0.5 if (2 * j == n) else 1.0
                s += bj * val * math.cos(2 * j * k * math.pi / n)
            
            weights[k] = (ck / n) * s
        return weights


# Convenience functions
def get_chebyshev_zeros(n: int) -> List[float]:
    """Get the n zeros of T_n(x).
    
    Args:
        n: Degree of Chebyshev polynomial
        
    Returns:
        List of n zeros in ascending order
    """
    return ChebyshevQuadrature().get_zeros(n)


def get_chebyshev_extrema(n: int) -> List[float]:
    """Get the n+1 extrema points of T_n(x).
    
    Args:
        n: Degree of Chebyshev polynomial
        
    Returns:
        List of n+1 extrema points
    """
    return ChebyshevQuadrature().get_extrema_points(n)


def gauss_chebyshev_integrate(f: Callable[[float], float], n: int = 64) -> float:
    """Integrate f(x) with weight 1/sqrt(1-x2) over [-1, 1].
    
    Args:
        f: Function to integrate
        n: Number of quadrature points (default: 64)
        
    Returns:
        Approximation of ?_{-1}^{1} f(x)/v(1-x2) dx
    """
    return ChebyshevQuadrature().gauss_chebyshev_quadrature(f, n)


def clenshaw_curtis_integrate(f: Callable[[float], float], n: int = 32) -> float:
    """Integrate f(x) over [-1, 1] using Clenshaw-Curtis quadrature.
    
    Automatically uses O(n2) method for small n and FFT-based O(n log n)
    method for large n (threshold: n >= 64).
    
    Args:
        f: Function to integrate
        n: Degree (uses n+1 points, default: 32)
        
    Returns:
        Approximation of ?_{-1}^{1} f(x) dx
    """
    return ChebyshevQuadrature().clenshaw_curtis_quadrature(f, n)


