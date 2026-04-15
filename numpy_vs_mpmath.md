# Analysis: Threshold-Based NumPy → mpmath Handover

## Summary of Findings

### ✅ Laguerre - YES, explicit threshold at n=200

  Found in OrthoPolyB/laguerre/utils.py:

  def stable_evaluation(n: int, x: float, alpha: float = 0.0, use_mp: bool = False):
      if use_mp or n > 200:  # ← THRESHOLD!
          val = evaluate_mp(n, x, alpha, 100)
          ...
      return GeneralizedLaguerrePolynomial(n, alpha).evaluate(x)

### ❌ Chebyshev - NO NumPy→mpmath handover for quadrature

- Has FFT_THRESHOLD = 64 but this is for algorithm selection (O(n²) vs O(n log n)), not precision switching
- Uses closed-form formulas (cos()), no automatic mpmath fallback

### ⚠️ Legendre - WARNING comments, but NO automatic handover

  From OrthoPolyB/legendre/core.py:

  """WARNING: For n > 60, these float coefficients may lose precision due to
  catastrophic cancellation. The coefficients grow very large (e.g., for n=50,
  coefficients exceed 10^14), leading to numerical instability."""

- Warns about n > 60 threshold
- Recommends using sympy_integration or mpmath_integration modules manually
- No automatic switching in core code

### ❌ Hermite - NO threshold, always uses mpmath (the problem!)

  Always uses high-precision by default — which is why we needed caching.

  ———

## Theoretical Analysis: Why Different Families Need Different Precision?

  | Family | Domain | Weight Function | Stability | Notes |
  |--------|--------|-----------------|-----------|-------|
  | Chebyshev | [-1, 1] | 1/√(1-x²) | ⭐⭐⭐ Most stable | Closed-form nodes: cos((2k-1)π/(2n)) |
  | Legendre | [-1, 1] | 1 | ⭐⭐ Moderate | Coefficients grow ~10¹⁴ at n=50 (catastrophic cancellation) |
  | Laguerre | [0, ∞) | x^α e^{-x} | ⭐⭐ Moderate | Condition number ~2√(nx) |
  | Hermite | (-∞, +∞) | e^{-x²} | ⭐ Least stable | Roots span wider range; higher condition numbers |

### Why Hermite Needs mpmath Most:

  1. Unbounded domain: Roots of H₆₄(x) span approximately [-12, +12]
  2. Rapidly growing coefficients: Hₙ has leading coefficient 2ⁿ
  3. Eigenvalue spread: Jacobi matrix eigenvalues are more widely separated than Chebyshev/Legendre

### Why Laguerre Has n=200 Threshold:

  The condition_estimate function in utils.py gives insight:

  def condition_estimate(n: int, x: float, alpha: float = 0.0) -> float:
      return 2 * math.sqrt(max(0, n * max(0, x))) + min(n * 0.5, 20)

  At n=200, the condition number becomes large enough that double precision (~16 digits) starts failing for certain x
  values.

  ———

## Recommendation

  Your instinct is correct: a uniform interface across all families would be ideal:

## Uniform API proposal

  GaussHermiteQuadrature(n=64, use_mpmath=False)  # Fast by default
  GaussHermiteQuadrature(n=200, use_mpmath=True)  # High precision when needed

## Or automatic with configurable threshold

  GaussHermiteQuadrature(n=64, auto_precision=True, mp_threshold=150)

  This would allow users to explicitly choose speed vs. accuracy tradeoffs consistently across all polynomial families.
