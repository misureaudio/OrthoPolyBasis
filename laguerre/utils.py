import math
import numpy as np
from typing import List, Tuple, Union


def to_sympy(n: int, alpha: float = 0.0):
    try:
        import sympy as sp
        x = sp.Symbol('x')
        return sp.laguerre(n, x) if alpha == 0 else sp.assoc_laguerre(n, alpha, x)
    except ImportError: raise ImportError('sympy required')


def evaluate_array(n: int, x_values, alpha: float = 0.0):
    from .polynomial import GeneralizedLaguerrePolynomial
    poly = GeneralizedLaguerrePolynomial(n, alpha)
    return poly.evaluate(np.asarray(x_values))


def generate_basis_matrix(max_degree: int, x_values, alpha: float = 0.0):
    x_arr = np.asarray(x_values, dtype=np.float64)
    V = np.zeros((len(x_arr), max_degree + 1))
    for j in range(max_degree + 1):
        V[:, j] = evaluate_array(j, x_arr, alpha)
    return V


def evaluate_mp(n: int, x, alpha: float = 0, prec: int = 50):
    import mpmath as mp
    mp.mp.dps = max(15, (prec-1)//3 + 1)
    x_m, a_m = mp.mpf(x), mp.mpf(alpha)
    if n == 0: return mp.mpf(1)
    p, c = mp.mpf(1), (a_m + 1 - x_m)
    for k in range(1, n):
        nxt = ((2*k + a_m + 1 - x_m)*c - (k + a_m)*p) / (k + 1)
        p, c = c, nxt
    return c


def stable_evaluation(n: int, x: float, alpha: float = 0.0, use_mp: bool = False):
    from .polynomial import GeneralizedLaguerrePolynomial
    if use_mp or n > 200:
        val = evaluate_mp(n, x, alpha, 100)
        try: return float(val)
        except OverflowError: return val
    return GeneralizedLaguerrePolynomial(n, alpha).evaluate(x)


def condition_estimate(n: int, x: float, alpha: float = 0.0) -> float:
    return 2 * math.sqrt(max(0, n * max(0, x))) + min(n * 0.5, 20)
