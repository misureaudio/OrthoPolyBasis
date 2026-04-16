# Quadratures

## Based on my exploration of the OrthoPolyB_np_mp library, I can provide you with a comprehensive guide on how to calculate integrals of functions on arbitrary real intervals.

## Summary

The library provides four families of orthogonal polynomials, each suited for different integration domains:

| Family | Natural Domain | Weight Function | Best For |
|--------|---------------|-----------------|----------|
| Legendre | [-1, 1] | None (uniform) | General finite intervals [a, b] |
| Chebyshev | [-1, 1] | 1/√(1-x²) | Functions with endpoint singularities |
| Hermite | (-∞, ∞) | e^(-x²) | Infinite intervals, Gaussian-weighted integrands |
| Laguerre | [0, ∞) | e^(-x) | Semi-infinite intervals |

  ———

## Integration on Arbitrary Finite Intervals [a, b]

### Using Legendre Polynomials (Recommended)

```Python
import sys
sys.path.insert(0, r'D:\Source\codex-dir\OrthoPolyB_np_mp')

from legendre import gauss_legendre, GaussLegendreQuadrature
import numpy as np

# Method 1: Using the convenience function with transformation
def integrate_on_interval(f, a, b, n=64):
    """Integrate f(x) from a to b using n-point Gauss-Legendre quadrature."""
    nodes, weights = gauss_legendre(n)
    return GaussLegendreQuadrature.integrate_transformed(f, a, b, nodes, weights)

# Example: ∫₀^π sin(x) dx = 2
result = integrate_on_interval(np.sin, 0, np.pi, n=32)
print(f"Result: {result}, Exact: 2.0, Error: {abs(result - 2.0):.2e}")

# Method 2: Manual transformation (more control)
def manual_transform_integration(f, a, b, n=64):
    """Manual interval transformation for educational purposes."""
    nodes, weights = gauss_legendre(n)

    # Transform from [-1, 1] to [a, b]: x_new = ((b-a)*x_old + (b+a))/2
    # dx_new = (b-a)/2 * dx_old
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0

    transformed_nodes = scale * nodes + shift
    scaled_weights = weights * scale

    return np.sum(scaled_weights * f(transformed_nodes))

# Same example with manual transformation
result_manual = manual_transform_integration(np.sin, 0, np.pi, n=32)
print(f"Manual result: {result_manual}")
```

### Using Chebyshev Polynomials (Clenshaw-Curtis Quadrature)

```Python
from chebyshev import clenshaw_curtis_integrate, get_chebyshev_extrema

def integrate_clenshaw_curtis(f, a, b, n=64):
    """Integrate using Clenshaw-Curtis quadrature on [a, b]."""
    # Get Chebyshev extrema (Clenshaw-Curtis nodes) in [-1, 1]
    nodes_std = np.array(get_chebyshev_extrema(n))

    # Transform to [a, b]
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    nodes = scale * nodes_std + shift

    # Compute Clenshaw-Curtis weights and integrate
    return clenshaw_curtis_integrate(lambda x: f(scale * x + shift), n=n) * scale

# Example: ∫_{-2}^{3} e^x dx = e³ - e^{-2} ≈ 19.854
result_cc = integrate_clenshaw_curtis(np.exp, -2, 3, n=64)
exact = np.exp(3) - np.exp(-2)
print(f"Clenshaw-Curtis: {result_cc}, Exact: {exact}, Error: {abs(result_cc - exact):.2e}")
```

  ———

## Integration on Infinite Intervals

### Semi-Infinite [0, ∞) with Laguerre Polynomials

```Python
from laguerre import LaguerreQuadrature

# For integrals of form: ∫₀^∞ f(x) e^{-x} dx
def integrate_laguerre(f, n=64):
    """Integrate f(x) * e^{-x} from 0 to ∞."""
    quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=False)
    return quad.integrate(f)

# Example: ∫₀^∞ x² e^{-x} dx = Γ(3) = 2! = 2
result_lag = integrate_laguerre(lambda x: x**2, n=64)
print(f"Laguerre integral: {result_lag}, Exact: 2.0, Error: {abs(result_lag - 2):.2e}")

# For integrals WITHOUT the e^{-x} weight, absorb it into f:
# ∫₀^∞ g(x) dx = ∫₀^∞ [g(x) * e^{x}] * e^{-x} dx
def integrate_laguerre_no_weight(g, n=64):
    """Integrate g(x) from 0 to ∞ (without weight)."""
    return integrate_laguerre(lambda x: g(x) * np.exp(x), n=n)

# Example: ∫₀^∞ e^{-2x} dx = 1/2
result_no_weight = integrate_laguerre_no_weight(lambda x: np.exp(-2*x), n=64)
print(f"Without weight: {result_no_weight}, Exact: 0.5, Error: {abs(result_no_weight - 0.5):.2e}")
Python
# Infinite (-∞, ∞) with Hermite Polynomials

from hermite import GaussHermiteQuadrature

# For integrals of form: ∫_{-∞}^{∞} f(x) e^{-x²} dx
def integrate_hermite(f, n=64):
    """Integrate f(x) * e^{-x²} from -∞ to ∞."""
    quad = GaussHermiteQuadrature(n=n, use_mpmath=False)
    return quad.integrate(f)

# Example: ∫_{-∞}^{∞} e^{-x²} dx = √π (with f(x)=1)
result_herm = integrate_hermite(lambda x: 1.0, n=64)
print(f"Hermite integral: {result_herm}, Exact: {np.sqrt(np.pi)}, Error: {abs(result_herm - np.sqrt(np.pi)):.2e}")

# For integrals WITHOUT the e^{-x²} weight:
def integrate_hermite_no_weight(g, n=64):
    """Integrate g(x) from -∞ to ∞ (without weight)."""
    return integrate_hermite(lambda x: g(x) * np.exp(x**2), n=n)

# Example: ∫_{-∞}^{∞} e^{-x²/2} dx = √(2π)
result_no_weight = integrate_hermite_no_weight(lambda x: np.exp(-x**2/2), n=64)
print(f"Without weight: {result_no_weight}, Exact: {np.sqrt(2*np.pi)}, Error: {abs(result_no_weight -
np.sqrt(2*np.pi)):.2e}")
```

  ———

## Complete Example: Unified Integration Interface

```Python
def smart_integrate(f, interval, n=64):
    """
    Automatically select quadrature method based on interval.

    Args:
        f: Function to integrate
        interval: Tuple (a, b) where a, b can be finite or ±np.inf
        n: Number of quadrature points

    Returns:
        Approximate integral value
    """
    a, b = interval

    # Finite interval [a, b] → Gauss-Legendre
    if np.isfinite(a) and np.isfinite(b):
        nodes, weights = gauss_legendre(n)
        return GaussLegendreQuadrature.integrate_transformed(f, a, b, nodes, weights)

    # Semi-infinite [0, ∞) → Gauss-Laguerre
    elif np.isfinite(a) and b == np.inf:
        if a == 0:
            quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=False)
            return quad.integrate(lambda x: f(x) * np.exp(x))
        else:
            # Shift to [0, ∞): let t = x - a
            quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=False)
            return quad.integrate(lambda t: f(t + a) * np.exp(t))

    # Semi-infinite (-∞, b] → Transform to [0, ∞)
    elif a == -np.inf and np.isfinite(b):
        quad = LaguerreQuadrature(n=n, alpha=0.0, use_mpmath=False)
        return quad.integrate(lambda t: f(b - t) * np.exp(t))

    # Infinite (-∞, ∞) → Gauss-Hermite
    else:
        quad = GaussHermiteQuadrature(n=n, use_mpmath=False)
        return quad.integrate(lambda x: f(x) * np.exp(x**2))

# Test cases
print("Testing smart integration:")
print(f"∫₀^π sin(x) dx = {smart_integrate(np.sin, (0, np.pi), 32):.10f} (exact: 2.0)")
print(f"∫_{-1}^{1} x² dx = {smart_integrate(lambda x: x**2, (-1, 1), 32):.10f} (exact: 0.667)")
print(f"∫₀^∞ e^{-x} dx = {smart_integrate(np.exp, (0, np.inf), 64):.10f} (exact: 1.0)")
```

  ———

## Key Notes

  1. Precision: Use use_mpmath=True for high-precision needs (especially Hermite with n ≥ 100)
  2. Caching: The library caches quadrature nodes/weights automatically
  3. Stability: Golub-Welsch method is recommended for Legendre (default)