"""
Gauss-Legendre Quadrature
=========================
Computes roots (nodes) and weights for Gauss-Legendre quadrature.

The n-point Gauss-Legendre quadrature exactly integrates polynomials of degree 2n-1:
    ?_{-1}^{1} f(x) dx ≈ sum_{i=1}^{n} w_i * f(x_i)

where x_i are roots of P_n(x) and weights w_i satisfy:
    w_i = 2 / ((1 - x_i²) [P_n'(x_i)]²)

Two algorithms provided:
1. Newton-Raphson: Iterative root-finding, good for moderate n
2. Golub-Welsch: Eigenvalue-based, numerically stable for high n
"""

from typing import List, Tuple, Union, Optional
import numpy as np
import warnings

# Module-level cache for quadrature nodes and weights
# Key: (n, use_mpmath), Value: (nodes_array, weights_array)
_gauss_legendre_cache = {}


def _validate_n(n: int) -> None:
    """Validate that n is a positive integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")


class GaussLegendreQuadrature:
    """
    Compute Gauss-Legendre quadrature nodes and weights.
    
    Provides two algorithms:
    - newton_raphson(): Iterative root-finding with good precision for moderate n
    - golub_welsch(): Eigenvalue-based, stable for high n (recommended)
    """
    
    @staticmethod
    def _evaluate_legendre(x: float, n: int) -> float:
        """Evaluate P_n(x) using three-term recurrence."""
        if n == 0:
            return 1.0
        if n == 1:
            return x
        
        p_prev2 = 1.0
        p_prev1 = x
        for k in range(2, n + 1):
            p_curr = ((2*k - 1) * x * p_prev1 - (k - 1) * p_prev2) / k
            p_prev2, p_prev1 = p_prev1, p_curr
        return p_curr
    
    @staticmethod
    def _evaluate_legendre_derivative(x: float, n: int) -> float:
        """
        Evaluate P_n'(x) using the identity:
            P_n'(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x² - 1)
        
        For x near ±1, use: P_n'(±1) = ±n(n+1)/2
        """
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0
        
        pn = GaussLegendreQuadrature._evaluate_legendre(x, n)
        pn_1 = GaussLegendreQuadrature._evaluate_legendre(x, n - 1)
        
        x2_minus_1 = x * x - 1.0
        if abs(x2_minus_1) < 1e-14:
            # Near endpoints: P_n'(±1) = ±n(n+1)/2
            sign = 1.0 if x > 0 else ((-1)**(n-1))
            return sign * n * (n + 1) / 2.0
        
        return n * (x * pn - pn_1) / x2_minus_1
    
    @staticmethod
    def newton_raphson(n: int, tol: float = 1e-15, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Legendre nodes and weights using Newton-Raphson iteration.
        
        Uses initial guesses based on cosine distribution of roots:
            x_i ≈ cos(π(4i-1)/(4n+2)) for i = 1,...,n
        
        Args:
            n: Number of quadrature points
            tol: Convergence tolerance for root-finding
            max_iter: Maximum Newton iterations per root
            
        Returns:
            (nodes, weights) where nodes are sorted in ascending order
            
        Example:
            >>> nodes, weights = GaussLegendreQuadrature.newton_raphson(3)
            >>> # Nodes: [-0.7745966692, 0.0, 0.7745966692]
            >>> # Weights: [0.5555555556, 0.8888888889, 0.5555555556]
        """
        _validate_n(n)
        
        nodes = np.zeros(n)
        weights = np.zeros(n)
        
        # Find each root using Newton-Raphson
        for i in range(n):
            # Initial guess: cosine distribution (roots are clustered near endpoints)
            theta = np.pi * (4 * (i + 1) - 1) / (4 * n + 2)
            x = np.cos(theta)
            
            # Newton-Raphson iteration
            for _ in range(max_iter):
                pn = GaussLegendreQuadrature._evaluate_legendre(x, n)
                dpn = GaussLegendreQuadrature._evaluate_legendre_derivative(x, n)
                
                if abs(dpn) < 1e-20:
                    break
                    
                dx = -pn / dpn
                x += dx
                
                if abs(dx) < tol:
                    break
            
            nodes[i] = x
            
            # Compute weight: w_i = 2 / ((1-x_i²) [P_n'(x_i)]²)
            dpn = GaussLegendreQuadrature._evaluate_legendre_derivative(x, n)
            weights[i] = 2.0 / ((1.0 - x * x) * dpn * dpn)
        
        # Sort nodes and weights in ascending order
        idx = np.argsort(nodes)
        nodes = nodes[idx]
        weights = weights[idx]
        
        return nodes, weights
    
    @staticmethod
    def golub_welsch(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Legendre nodes and weights using the Golub-Welsch algorithm.
        
        This eigenvalue-based method is numerically stable for high n.
        The nodes are eigenvalues of the Jacobi matrix:
            J = diag(0, 1/√3, 2/√8, ...) ⊗ offdiag(1/√3, √8/4, ...)
        
        Args:
            n: Number of quadrature points
            
        Returns:
            (nodes, weights) where nodes are sorted in ascending order
            
        Note:
            This method is recommended for n > 50 due to better numerical stability.
            
        Example:
            >>> nodes, weights = GaussLegendreQuadrature.golub_welsch(3)
        """
        _validate_n(n)
        
        if n == 1:
            return np.array([0.0]), np.array([2.0])
        
        # Build the Jacobi matrix for Legendre polynomials
        # Diagonal elements: all zero for Legendre
        # Off-diagonal: b_k = k / sqrt(4*k^2 - 1) for k = 1, ..., n-1
        
        k = np.arange(1, n)
        beta = k / np.sqrt(4.0 * k * k - 1.0)
        
        # Construct symmetric tridiagonal matrix
        J = np.zeros((n, n))
        np.fill_diagonal(J, 0.0)  # Diagonal is zero for Legendre
        for i in range(n - 1):
            J[i, i+1] = beta[i]
            J[i+1, i] = beta[i]
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(J)
        
        # Nodes are eigenvalues
        nodes = eigenvalues
        
        # Weights: w_i = 2 * [v_{i,0}]² where v_{i,0} is first component of i-th eigenvector
        weights = 2.0 * (eigenvectors[0, :] ** 2)
        
        # Ensure sorted order (eigh should return sorted eigenvalues)
        idx = np.argsort(nodes)
        nodes = nodes[idx]
        weights = weights[idx]
        
        return nodes, weights
    
    @staticmethod
    def compute(n: int, method: str = "golub_welsch") -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Legendre quadrature nodes and weights.
        
        Args:
            n: Number of quadrature points
            method: Algorithm to use - "newton_raphson" or "golub_welsch"
                   Default is "golub_welsch" (recommended for stability)
            
        Returns:
            (nodes, weights) numpy arrays
            
        Raises:
            ValueError: If method is not recognized
        """
        _validate_n(n)
        
        if method == "newton_raphson":
            return GaussLegendreQuadrature.newton_raphson(n)
        elif method == "golub_welsch":
            return GaussLegendreQuadrature.golub_welsch(n)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'newton_raphson' or 'golub_welsch'")
    
    @staticmethod
    def integrate(f, nodes: np.ndarray, weights: np.ndarray) -> float:
        """
        Integrate function f over [-1, 1] using precomputed quadrature.
        
        Args:
            f: Function to integrate (callable)
            nodes: Quadrature nodes
            weights: Quadrature weights
            
        Returns:
            Approximation of ?_{-1}^{1} f(x) dx
        """
        return np.sum(weights * f(nodes))
    
    @staticmethod  
    def integrate_transformed(f, a: float, b: float, nodes: np.ndarray, weights: np.ndarray) -> float:
        """
        Integrate function f over [a, b] using quadrature on [-1, 1].
        
        Uses change of variables: x = ((b-a)t + (b+a))/2
        ?_{a}^{b} f(x) dx = (b-a)/2 * ?_{-1}^{1} f(((b-a)t+(b+a))/2) dt
        
        Args:
            f: Function to integrate
            a, b: Integration limits
            nodes: Quadrature nodes on [-1, 1]
            weights: Quadrature weights
            
        Returns:
            Approximation of ?_{a}^{b} f(x) dx
        """
        scale = (b - a) / 2.0
        shift = (b + a) / 2.0
        transformed_nodes = scale * nodes + shift
        return scale * np.sum(weights * f(transformed_nodes))


# Convenience functions
def gauss_legendre(n: int, method: str = "golub_welsch") -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute Gauss-Legendre quadrature.
    
    Args:
        n: Number of quadrature points
        method: Algorithm - "newton_raphson" or "golub_welsch" (default)
        
    Returns:
        (nodes, weights) tuple
    """
    return GaussLegendreQuadrature.compute(n, method)


def gauss_legendre_newton(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute using Newton-Raphson method."""
    return GaussLegendreQuadrature.newton_raphson(n)


def gauss_legendre_golub_welsch(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute using Golub-Welsch eigenvalue method."""
    return GaussLegendreQuadrature.golub_welsch(n)

class LegendreQuadrature:
    """
    Gauss-Legendre quadrature with automatic NumPy/mpmath selection.
    
    Provides cached quadrature nodes and weights with optional high-precision
    mpmath computation for very large n (>= 200).
    
    Example:
        >>> quad = LegendreQuadrature(64)
        >>> nodes, weights = quad.nodes, quad.weights
        >>> result = np.sum(weights * f(nodes))
    """
    
    def __init__(self, n: int, use_mpmath: bool = False):
        """
        Initialize quadrature with n points.
        
        Args:
            n: Number of quadrature points
            use_mpmath: If True, use high-precision mpmath library.
                       Recommended for n >= 200. Default: False.
        
        Notes:
            - NumPy Golub-Welsch is stable up to n ~ 150-200
            - For n >= 200, consider use_mpmath=True for better accuracy
        """
        self.n = n
        self._use_mpmath = use_mpmath
        
        # Warn if using NumPy at very high degrees
        if not use_mpmath and n >= 200:
            warnings.warn(
                f"Legendre quadrature with n={n} using NumPy (double precision). "
                "For n >= 200, consider use_mpmath=True for better accuracy.",
                UserWarning,
                stacklevel=2
            )
        
        # Check module-level cache first
        cache_key = (n, use_mpmath)
        if cache_key not in _gauss_legendre_cache:
            if use_mpmath:
                # High-precision path using mpmath
                _gauss_legendre_cache[cache_key] = self._compute_mpmath(n)
            else:
                # Fast NumPy Golub-Welsch (already stable for Legendre)
                _gauss_legendre_cache[cache_key] = GaussLegendreQuadrature.golub_welsch(n)
        
        # Load from cache
        self._nodes, self._weights = _gauss_legendre_cache[cache_key]
    
    def _compute_mpmath(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute quadrature using mpmath high-precision arithmetic.
        Uses Golub-Welsch algorithm with arbitrary precision.
        """
        import mpmath as mp
        
        # Set high precision
        mp.mp.dps = 80  # 80 decimal places
        
        # Build Jacobi matrix in mpmath
        diag = [mp.mpf(0)] * n
        off_diag = []
        for k in range(1, n):
            beta = mp.mpf(k) / mp.sqrt(mp.mpf(4) * k * k - mp.mpf(1))
            off_diag.append(beta)
        
        # Construct matrix and compute eigenvalues
        J = mp.matrix(n, n)
        for i in range(n):
            J[i, i] = diag[i]
            if i < n - 1:
                J[i, i+1] = off_diag[i]
                J[i+1, i] = off_diag[i]
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = mp.eig(J, left=False)
        
        # Sort by eigenvalue
        sorted_indices = sorted(range(n), key=lambda i: eigenvals[i])
        nodes = [eigenvals[i] for i in sorted_indices]
        # First component of each eigenvector (column-major: eigenvecs[0, i])
        weights = [mp.mpf(2) * eigenvecs[0, i]**2 for i in sorted_indices]
        
        # Convert to NumPy
        return np.array([float(x) for x in nodes]), np.array([float(w) for w in weights])
    
    @property
    def nodes(self) -> np.ndarray:
        """Quadrature nodes (roots of P_n)."""
        return self._nodes.copy()
    
    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights."""
        return self._weights.copy()
    
    def integrate(self, f, vectorized: bool = True) -> float:
        """
        Compute integral of f(x) from -1 to +1.
        
        Args:
            f: Function to integrate
            vectorized: If True, f accepts array input; if False, called point-by-point
        
        Returns:
            Approximate value of ?_{-1}^{+1} f(x) dx
        """
        if vectorized:
            f_values = f(self._nodes)
        else:
            f_values = np.array([f(x) for x in self._nodes])
        
        return float(np.sum(self._weights * f_values))
