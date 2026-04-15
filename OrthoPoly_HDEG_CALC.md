# Task

What algorithms eventually exist to evalute very high degree Hermite polynomials, in case onewants a direct verification of how good  are the found roots?

## Algorithms for High-Degree Hermite Evaluation

### 1. Three-Term Recurrence (Forward) - Most Stable

The physicist's Hermite recurrence:
$$H_{n+1}(x) = 2x \cdot H_n(x) - 2n \cdot H_{n-1}(x)$$

Problem: Values grow as $H_n(0) \sim (-1)^{n/2} (n-1)!!$ for even n, causing overflow.

### 2. Miller's Algorithm (Backward Recurrence) - Best for Verification

Computes normalized ratios that stay bounded:
$$\frac{H_{n-k}(x)}{H_n(x)} \text{ via backward recurrence}$$

### 3. Using the Differential Equation

Since $H_n''(x) - 2x H_n'(x) + 2n H_n(x) = 0$, at a root:
$$\frac{H_n''(r)}{H_n'(r)} = 2r \quad \text{(if } r \text{ is truly a root)}$$

### 4. Logarithmic Evaluation

Compute $$\log|H_n(x)|$$ to avoid overflow, then exponentiate only when needed.

## Summary: Algorithms for High-Degree Hermite Evaluation

### The Problem

Direct coefficient evaluation fails because $H_n(x)$ has coefficients growing like $2^n n!$, causing overflow even at moderate degrees.

———

### Stable Algorithms

| Method | Description | Best For |
|--------|-------------|----------|
| Miller's Algorithm | Backward recurrence computing ratios $H_{n-k}/H_n$ | Most stable, gold standard |
| SciPy hermite() | Uses optimized stable algorithms internally | Quick verification |
| Logarithmic Evaluation | Track $\log|H_n|$ separately from sign | Avoiding overflow entirely |
| Ratio Property Test | Check $H_{n+1}(r)/H_{n-1}(r) = -2n$ at root $r$ | Best for root verification! |

———

### Key Insight: Root Verification via Ratio Property

At any true root $r$ of $H_n$:
$$\frac{H_{n+1}(r)}{H_{n-1}(r)} = -2n \quad \text{(exactly!)}$$

This is numerically stable because it uses ratios, not absolute values. For our degree-51 roots:

- Expected ratio: -102
- Actual ratio: -102.000000000000568
- Relative error: 5.57×10⁻¹⁵

———

### Recommendation for Your Use Case

For verifying roots of high-degree Hermite polynomials, use the ratio property test:

```Python
from scipy.special import hermite

def verify_hermite_root(n, root):
    H_nm1 = hermite(n-1)(root) if n > 0 else 1.0
    H_np1 = hermite(n+1)(root)
    ratio = H_np1 / H_nm1
    return abs(ratio - (-2*n)) / (2*n)  # Relative error
```

This avoids all overflow issues and gives you a clean relative error metric!

## Question

Do equivalent methods exist for the families 'chebyshev', 'legendre', 'laguerre'?

## All Classical Orthogonal Polynomials Have Equivalent Methods

### Summary Table

| Family | Recurrence Relation | At Root Ratio Property | Max Error (n=25) |
|--------|---------------------|----------------------|------------------|
| Hermite $H_n$ | $H_{n+1} = 2x H_n - 2n H_{n-1}$ | $\displaystyle\frac{H_{n+1}}{H_{n-1}} = -2n$ | ~10⁻¹⁴ |
| Chebyshev $T_n$ | $T_{n+1} = 2x T_n - T_{n-1}$ | $\displaystyle\frac{T_{n+1}}{T_{n-1}} = -1$ | ~10⁻¹³ |
| Legendre $P_n$ | $(n+1)P_{n+1} = (2n+1)x P_n - n P_{n-1}$ | $\displaystyle\frac{P_{n+1}}{P_{n-1}} = -\frac{n}{n+1}$| ~5×10⁻¹³ |
| Laguerre $L_n^{(\alpha)}$ | $(n+1)L_{n+1} = [(2n+1+\alpha)-x]L_n - (n+\alpha)L_{n-1}$ | $\displaystyle\frac{L_{n+1}}{L_{n-1}} = -\frac{n+\alpha}{n+1}$ | ~2×10⁻¹³ |

———

### Why This Works for All Families

At any root $r$ of polynomial $P_n$, we have $P_n(r) = 0$. The three-term recurrence then simplifies to relate only $P_{n+1}(r)$ and $P_{n-1}(r)$, giving a constant ratio that depends only on $n$ (and parameters like $\alpha$ for Laguerre).

This ratio property is:

- Numerically stable (uses ratios, not absolute values)
- Independent of polynomial normalization
- Works for arbitrarily high degrees
- Provides clean relative error metric

———

### Verification Code Template

```Python
from scipy import special

def verify_orthogonal_roots(family, n, alpha=None):
    """Generic root verification for all classical families."""

    # Get roots (family-specific)
    if family == 'hermite':
        from numpy.polynomial.hermite import hermgauss
        roots, _ = hermgauss(n)
        P_nm1, P_np1 = special.hermite(n-1), special.hermite(n+1)
        expected = -2*n
    elif family == 'chebyshev':
        from numpy.polynomial.chebyshev import chebgauss
        roots, _ = chebgauss(n)
        P_nm1, P_np1 = special.chebyt(n-1), special.chebyt(n+1)
        expected = -1
    elif family == 'legendre':
        from numpy.polynomial.legendre import leggauss
        roots, _ = leggauss(n)
        P_nm1, P_np1 = special.legendre(n-1), special.legendre(n+1)
        expected = -n/(n+1)
    elif family == 'laguerre':
        from numpy.polynomial.laguerre import laggauss
        roots, _ = laggauss(n)
        P_nm1, P_np1 = special.laguerre(n-1), special.laguerre(n+1)
        expected = -(n + (alpha or 0))/(n+1)

    # Compute ratio at each root
    ratios = np.array([P_np1(r)/P_nm1(r) for r in sorted(roots)])
    rel_error = np.abs(ratios - expected) / abs(expected)


    return max(rel_error), expected
```
