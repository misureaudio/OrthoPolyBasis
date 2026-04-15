import math
import numpy as np
from typing import Union, List, Tuple


class GeneralizedLaguerrePolynomial:
    def __init__(self, n: int, alpha: float = 0.0):
        if n < 0: raise ValueError('Degree must be non-negative')
        if alpha <= -1: raise ValueError('Alpha must be > -1')
        self.n, self.alpha = n, alpha
        self._coeffs = None

    def get_coefficients(self) -> List[float]:
        if self._coeffs is not None: return self._coeffs.copy()
        coeffs = []
        for k in range(self.n + 1):
            # Standard Formula: a_k = [(-1)^k / k!] * binom(n + alpha, n - k)
            log_c = (math.lgamma(self.n + self.alpha + 1) - math.lgamma(k + self.alpha + 1) - 
                     math.lgamma(self.n - k + 1) - math.lgamma(k + 1))
            coeffs.append(math.exp(log_c) * ((-1)**k))
        self._coeffs = coeffs
        return self._coeffs.copy()

    def evaluate_with_derivative(self, x: Union[float, np.ndarray], use_mpmath: bool = False) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Evaluate polynomial and its derivative.
        
        Args:
            x: Point(s) to evaluate at
            use_mpmath: If True, use high-precision mpmath for evaluation.
                       Recommended for n >= 200 or extreme accuracy needs.
        """
        is_arr = isinstance(x, np.ndarray)
        
        # Route through stable_evaluation if mpmath requested
        if use_mpmath:
            from .utils import evaluate_mp
            if is_arr:
                vals = np.array([float(evaluate_mp(self.n, float(xi), self.alpha, 100)) for xi in x])
            else:
                val = evaluate_mp(self.n, float(x), self.alpha, 100)
                try:
                    vals = float(val)
                except OverflowError:
                    vals = val
            # Derivative not available in mpmath path, return zeros
            return vals, (np.zeros_like(x) if is_arr else 0.0)
        
        # Standard NumPy three-term recurrence
        if self.n == 0:
            return (np.ones_like(x) if is_arr else 1.0), (np.zeros_like(x) if is_arr else 0.0)

        l_p, d_p = (np.ones_like(x) if is_arr else 1.0), (np.zeros_like(x) if is_arr else 0.0)
        l_c, d_c = (self.alpha + 1 - x), (np.full_like(x, -1.0) if is_arr else -1.0)

        if self.n == 1: return l_c, d_c

        for k in range(1, self.n):
            l_n = ((2*k + self.alpha + 1 - x)*l_c - (k + self.alpha)*l_p) / (k + 1)
            d_n = ((2*k + self.alpha + 1 - x)*d_c - l_c - (k + self.alpha)*d_p) / (k + 1)
            l_p, l_c = l_c, l_n
            d_p, d_c = d_c, d_n
        return l_c, d_c

    def evaluate(self, x, use_mpmath: bool = False):
        """
        Evaluate polynomial at point(s).
        
        Args:
            x: Point(s) to evaluate at
            use_mpmath: If True, use high-precision mpmath for evaluation.
                       Recommended for n >= 200 or extreme accuracy needs.
        """
        v, _ = self.evaluate_with_derivative(x, use_mpmath)
        return v

    def __call__(self, x): return self.evaluate(x)

    @property
    def degree(self): return self.n


class LaguerrePolynomial(GeneralizedLaguerrePolynomial):
    def __init__(self, n: int):
        super().__init__(n, alpha=0.0)
