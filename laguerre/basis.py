import math
import numpy as np
import warnings
from typing import Callable, List, Tuple
from .polynomial import GeneralizedLaguerrePolynomial

# Module-level cache for quadrature nodes and weights
# Key: (n, alpha, use_mpmath), Value: (nodes_array, weights_array)
_laguerre_quadrature_cache = {}


class GeneralizedLaguerreBasis:
    def __init__(self, max_degree: int, alpha: float = 0.0):
        if max_degree < 0: raise ValueError('max_degree must be non-negative')
        if alpha <= -1: raise ValueError('alpha must be > -1')
        self.max_degree, self.alpha = max_degree, alpha
        self._polys = [GeneralizedLaguerrePolynomial(n, alpha) for n in range(max_degree + 1)]
        self._quad_cache = {}

    def __getitem__(self, n): return self._polys[n]
    def __len__(self): return len(self._polys)
    def __iter__(self): return iter(self._polys)

    def weight_function(self, x: float) -> float:
        if x < 0: return 0.0
        return math.pow(x, self.alpha) * math.exp(-x) if x > 0 else (1.0 if self.alpha == 0 else 0.0)

    def norm_squared(self, n: int) -> float:
        return math.exp(math.lgamma(n + self.alpha + 1) - math.lgamma(n + 1))

    def _gauss_quadrature(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Golub-Welsch Algorithm: eigenvalues of Jacobi matrix are roots
        i = np.arange(n)
        diag = 2 * i + self.alpha + 1
        off = np.sqrt(np.arange(1, n) * (np.arange(1, n) + self.alpha))
        J = np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)
        roots, evecs = np.linalg.eigh(J)
        wts = (evecs[0, :]**2) * math.gamma(self.alpha + 1)
        return roots, wts

    def inner_product(self, f, g=None):
        if g is None: g = f
        N = max(self.max_degree + 1, 64)

        # Simple, safe instance-level cache
        if not hasattr(self, '_quad_cache'):
            self._quad_cache = {}

        if N not in self._quad_cache:
            self._quad_cache[N] = self._gauss_quadrature(N)

        pts, wts = self._quad_cache[N]
        return sum(f(x) * g(x) * w for x, w in zip(pts, wts))

    def project(self, f: Callable) -> List[float]:
        return [self.inner_product(f, P.evaluate) / self.norm_squared(n) for n, P in enumerate(self._polys)]

    def approximate(self, f: Callable) -> Callable:
        coeffs = self.project(f)
        return lambda x: sum(c * P.evaluate(x) for c, P in zip(coeffs, self._polys))


class LaguerreBasis(GeneralizedLaguerreBasis):
    def __init__(self, max_degree: int):
        super().__init__(max_degree, alpha=0.0)


def compute_roots(n: int, alpha: float = 0.0) -> List[float]:
    roots, _ = GeneralizedLaguerreBasis(n, alpha)._gauss_quadrature(n)
    return roots


def gauss_quadrature_weights(n: int, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    return GeneralizedLaguerreBasis(n, alpha)._gauss_quadrature(n)


def function_projection(f, max_degree: int, alpha: float = 0.0):
    return GeneralizedLaguerreBasis(max_degree, alpha).project(f)


def function_approximation(f, max_degree: int, alpha: float = 0.0):
    return GeneralizedLaguerreBasis(max_degree, alpha).approximate(f)

class LaguerreQuadrature:
    """
    Gauss-Laguerre quadrature with automatic NumPy/mpmath selection.
    
    Provides cached quadrature nodes and weights for integrals of the form:
        ?_{0}^{?} f(x) * x^? * e^{-x} dx ? ? w_i * f(x_i)
    
    Example:
        >>> quad = LaguerreQuadrature(64, alpha=0)
        >>> nodes, weights = quad.nodes, quad.weights
        >>> result = np.sum(weights * f(nodes))
    """
    
    def __init__(self, n: int, alpha: float = 0.0, use_mpmath: bool = False):
        """
        Initialize quadrature with n points.
        
        Args:
            n: Number of quadrature points
            alpha: Parameter ? > -1 for generalized Laguerre L_n^(?)
            use_mpmath: If True, use high-precision mpmath library.
                       Recommended for n >= 200. Default: False.
        
        Notes:
            - NumPy Golub-Welsch is stable up to n ~ 150-200
            - For n >= 200, consider use_mpmath=True for better accuracy
        """
        if alpha <= -1:
            raise ValueError("alpha must be > -1")
        
        self.n = n
        self.alpha = alpha
        self._use_mpmath = use_mpmath
        
        # Warn if using NumPy at high degrees
        if not use_mpmath and n >= 200:
            warnings.warn(
                f"Laguerre quadrature with n={n} using NumPy (double precision). "
                "For n >= 200, consider use_mpmath=True for better accuracy.",
                UserWarning,
                stacklevel=2
            )
        
        # Check module-level cache first
        cache_key = (n, alpha, use_mpmath)
        if cache_key not in _laguerre_quadrature_cache:
            if use_mpmath:
                _laguerre_quadrature_cache[cache_key] = self._compute_golub_welsch_mp(n, alpha)
            else:
                _laguerre_quadrature_cache[cache_key] = self._compute_golub_welsch_np(n, alpha)
        
        # Load from cache
        self._nodes, self._weights = _laguerre_quadrature_cache[cache_key]
    
    def _compute_golub_welsch_np(self, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute quadrature using NumPy Golub-Welsch algorithm.
        Fast but limited to double precision (~16 digits).
        """
        # Build Jacobi matrix for generalized Laguerre polynomials
        # Three-term recurrence (monic form):
        #   p_{k+1}(x) = (x - a_k)*p_k(x) - b_k*p_{k-1}(x)
        # where:
        #   a_k = 2*k + alpha + 1
        #   b_k = k*(k + alpha) for k >= 1, b_0 = gamma(alpha+1)
        
        diag = np.array([2.0 * k + alpha + 1.0 for k in range(n)])
        off_diag = np.sqrt(np.arange(1, n) * (np.arange(1, n) + alpha))
        
        # Construct symmetric tridiagonal Jacobi matrix
        J = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        # Eigenvalues are the quadrature nodes (roots)
        eigenvalues, eigenvectors = np.linalg.eigh(J)
        
        # Weights: w_i = gamma(alpha+1) * v_{0,i}^2
        weights = math.gamma(alpha + 1) * (eigenvectors[0, :] ** 2)
        
        return eigenvalues, weights
    
    def _compute_golub_welsch_mp(self, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute quadrature using mpmath high-precision arithmetic.
        Slower but accurate for very large n or extreme precision needs.
        """
        import mpmath as mp
        
        # Set high precision
        mp.mp.dps = 80  # 80 decimal places
        
        alpha_mp = mp.mpf(alpha)
        
        # Build Jacobi matrix in mpmath
        diag = [mp.mpf(2) * k + alpha_mp + mp.mpf(1) for k in range(n)]
        off_diag = []
        for k in range(1, n):
            val = mp.sqrt(mp.mpf(k) * (mp.mpf(k) + alpha_mp))
            off_diag.append(val)
        
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
        weights = [mp.gamma(alpha_mp + 1) * eigenvecs[0, i]**2 for i in sorted_indices]
        
        # Convert to NumPy
        return np.array([float(x) for x in nodes]), np.array([float(w) for w in weights])
    
    @property
    def nodes(self) -> np.ndarray:
        """Quadrature nodes (roots of L_n^(?))."""
        return self._nodes.copy()
    
    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights."""
        return self._weights.copy()
    
    def integrate(self, f, vectorized: bool = True) -> float:
        """
        Compute integral of f(x) * x^? * e^{-x} from 0 to +?.
        
        Args:
            f: Function to integrate
            vectorized: If True, f accepts array input; if False, called point-by-point
        
        Returns:
            Approximate value of ?_{0}^{?} f(x) * x^? * e^{-x} dx
        """
        if vectorized:
            f_values = f(self._nodes)
        else:
            f_values = np.array([f(x) for x in self._nodes])
        
        return float(np.sum(self._weights * f_values))
