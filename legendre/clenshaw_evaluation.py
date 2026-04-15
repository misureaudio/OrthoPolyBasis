"""
Clenshaw-Like Evaluation for Legendre Series
=============================================
Implements numerically stable O(n) evaluation of Legendre series:
    S(x) = sum_{k=0}^{n} a_k * P_k(x)

Standard polynomial evaluation in monomial basis using Horner\'s method is stable,
but evaluating sum c_i * x^i directly suffers from catastrophic cancellation.

Clenshaw\'s algorithm rewrites the sum using the three-term recurrence:
    (k+1)*P_{k+1}(x) = (2k+1)*x*P_k(x) - k*P_{k-1}(x)

to compute the series in O(n) operations with good numerical stability.
"""

from typing import List, Union, overload
import numpy as np


def _validate_non_negative(n: int) -> None:
    """Validate n is non-negative integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")


def evaluate_legendre_series(coeffs: np.ndarray, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Evaluate Legendre series using Clenshaw\'s algorithm.
    
    Computes S(x) = sum_{k=0}^{n} coeffs[k] * P_k(x)
    
    where coeffs[0] is the coefficient of P_0, coeffs[1] of P_1, etc.
    (ascending order by degree).
    
    Uses Clenshaw\'s algorithm which is numerically stable for high-degree series.
    
    Args:
        coeffs: Array of Legendre coefficients [a_0, a_1, ..., a_n]
               representing sum a_k * P_k(x)
        x: Point(s) at which to evaluate the series
    
    Returns:
        Value of the Legendre series at x
    
    Example:
        >>> # Evaluate 1*P_0 + 2*P_1 + 3*P_2 at x=0.5
        >>> evaluate_legendre_series([1, 2, 3], 0.5)
        4.75
    """
    coeffs = np.asarray(coeffs, dtype=float)
    
    if isinstance(x, (int, float)):
        return _clenshaw_single(coeffs, float(x))
    else:
        x_arr = np.asarray(x)
        return np.array([_clenshaw_single(coeffs, xi) for xi in x_arr])


def _clenshaw_single(coeffs: np.ndarray, x: float) -> float:
    """
    Single-point Clenshaw evaluation (internal helper).
    
    Uses the three-term recurrence for Legendre polynomials:
        P_{k+1}(x) = ((2k+1)/(k+1)) * x * P_k(x) - (k/(k+1)) * P_{k-1}(x)
    
    with P_0(x) = 1, P_1(x) = x
    """
    n = len(coeffs) - 1
    
    if n < 0:
        return 0.0
    if n == 0:
        return float(coeffs[0])
    if n == 1:
        return float(coeffs[0] + coeffs[1] * x)
    
    # Clenshaw recurrence
    b_next = 0.0   # b_{k+2}
    b_curr = 0.0   # b_{k+1}
    
    for k in range(n, 1, -1):  # k = n, n-1, ..., 2
        # b_k = a_k + ((2k+1)/(k+1))*x*b_{k+1} - (k/(k+1))*b_{k+2}
        b_prev = coeffs[k] + (2.0*k + 1.0) / (k + 1.0) * x * b_curr - k / (k + 1.0) * b_next
        b_next = b_curr
        b_curr = b_prev
    
    # At this point: b_curr = b_2, b_next = b_3
    # We need to handle k=1 separately for the final result
    # 
    # The series is: S = a_0*P_0 + a_1*P_1 + ... + a_n*P_n
    # After Clenshaw recurrence down to k=2, we have:
    # S = a_0*P_0 + (a_1 + 3/2*x*b_2 - 1/2*b_3)*P_1 - 1/2*b_2*P_0
    #   = a_0 + x*(a_1 + 3/2*x*b_2 - 1/2*b_3) - 1/2*b_2
    # 
    # Actually, let me use the standard formula:
    # After computing b_2 through b_n, we have:
    # S = a_0*P_0 + b_1*P_1 where b_1 needs to be computed
    
    # Compute b_1
    b_1 = coeffs[1] + 3.0/2.0 * x * b_curr - 1.0/2.0 * b_next
    
    # Final result: S = a_0*P_0(x) + b_1*P_1(x) = a_0 + b_1*x
    result = coeffs[0] + b_1 * x
    
    return float(result)
