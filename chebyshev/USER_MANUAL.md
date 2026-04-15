# Chebyshev Polynomials Library - User Manual

## Overview

The `chebyshev` library provides a Python implementation for computing and working with Chebyshev polynomials — a family of orthogonal polynomials widely used in:

- **Numerical analysis** — polynomial interpolation, function approximation, spectral methods
- **Scientific computing** — solving differential equations, least-squares fitting
- **Signal processing** — filter design, frequency response analysis
- **Approximation theory** — minimax approximations (Chebyshev equioscillation theorem)

---

## Installation

### Requirements

```python
# Core dependencies (already included in the package)
numpy      # For FFT-based Clenshaw-Curtis quadrature
scipy      # Optional: for additional numerical methods
mpmath     # Optional: for high-precision arithmetic with large degrees
```

### Quick Start

```python
from chebyshev import (
    ChebyshevQuadrature,
    get_chebyshev_zeros,
    get_chebyshev_extrema,
    gauss_chebyshev_integrate,
    clenshaw_curtis_integrate
)
```

---

## Core Polynomial Operations

### `chebyshev_derivative_stable(n, x)` — Derivative of Tₙ(x)

Calculates the derivative of the n-th Chebyshev polynomial at point x using stable recurrence.

```python
from chebyshev import chebyshev_derivative_stable

# Calculate T_3'(x) at various points
for x in [-1, -0.5, 0, 0.5, 1]:
    deriv = chebyshev_derivative_stable(3, x)
    print(f"T_3'({x}) = {deriv}")
```

**Parameters:**
- `n` (int): Degree of Chebyshev polynomial (must be ≥ 0)
- `x` (float): Point at which to evaluate the derivative

**Returns:** Derivative value Tₙ'(x)

---

### `chebyshev_integral_stable(n, x)` — Integral of Tₙ(x)

Calculates the indefinite integral of the n-th Chebyshev polynomial at point x.

```python
from chebyshev import chebyshev_integral_stable

# Calculate ∫T_2(x) dx from 0 to x
for x in [0, 0.5, 1]:
    integral = chebyshev_integral_stable(2, x)
    print(f"∫₀ˣ T₂(t) dt at x={x} = {integral}")
```

**Parameters:**
- `n` (int): Degree of Chebyshev polynomial (must be ≥ 0)
- `x` (float): Point at which to evaluate the integral

**Returns:** Integral value ∫₀ˣ Tₙ(t) dt

---

## Quadrature and Roots Computation

### `ChebyshevQuadrature` — Main Interface for Nodes and Integration

Provides comprehensive functionality for computing Chebyshev nodes and performing numerical integration.

#### Constructor
```python
from chebyshev import ChebyshevQuadrature

q = ChebyshevQuadrature()
```

---

### `get_zeros(n)` — Gauss-Chebyshev Nodes (Zeros of Tₙ)

Computes the n zeros of Tₙ(x), which are the optimal nodes for Gauss-Chebyshev quadrature.

```python
from chebyshev import ChebyshevQuadrature

q = ChebyshevQuadrature()
zeros = q.get_zeros(4)
print(f"Zeros of T_4(x): {zeros}")
# Output: [-0.9238795, -0.3826834, 0.3826834, 0.9238795]
```

**Parameters:**
- `n` (int): Degree of Chebyshev polynomial (number of zeros)

**Returns:** List of n zeros in ascending order

**Mathematical Formula:** xₖ = cos((2k-1)π/(2n)) for k=1..n

---

### `get_extrema_points(n)` — Clenshaw-Curtis Nodes (Extrema of Tₙ)

Computes the n+1 extrema points of Tₙ(x), where |Tₙ| = 1. These are used for Clenshaw-Curtis quadrature.

```python
from chebyshev import ChebyshevQuadrature

q = ChebyshevQuadrature()
extrema = q.get_extrema_points(3)
print(f"Extrema of T_3(x): {extrema}")
# Output: [1.0, 0.5, -0.5, -1.0]
```

**Parameters:**
- `n` (int): Degree of Chebyshev polynomial

**Returns:** List of n+1 extrema points in descending order (from 1 to -1)

**Mathematical Formula:** xₖ = cos(kπ/n) for k=0..n

---

### `get_gauss_chebyshev_weights(n)` — Gauss-Chebyshev Quadrature Weights

Returns the weights for Gauss-Chebyshev quadrature.

```python
from chebyshev import ChebyshevQuadrature
import math

q = ChebyshevQuadrature()
weights = q.get_gauss_chebyshev_weights(4)
print(f"Gauss-Chebyshev weights (n=4): {weights}")
# Output: [0.7853981633974483, 0.7853981633974483, ...]
# All equal to π/4 ≈ 0.7854
```

**Parameters:**
- `n` (int): Number of quadrature points

**Returns:** List of n weights (all equal to π/n)

---

### `gauss_chebyshev_quadrature(f, n)` — Weighted Integration

Approximates the weighted integral:
```
∫_{-1}^{1} f(x) / √(1-x²) dx ≈ (π/n) * Σ_{k=1}^{n} f(xₖ)
```

```python
from chebyshev import ChebyshevQuadrature
import math

q = ChebyshevQuadrature()

# Integrate x² with weight 1/√(1-x²) over [-1, 1]
result = q.gauss_chebyshev_quadrature(lambda x: x**2, n=64)
print(f"∫_{-1}^{1} x²/√(1-x²) dx ≈ {result}")
# Exact value is π/2 ≈ 1.5708
```

**Parameters:**
- `f` (callable): Function to integrate
- `n` (int): Number of quadrature points (higher = more accurate)

**Returns:** Approximation of the weighted integral

---

### `clenshaw_curtis_quadrature(f, n)` — Standard Integration

Approximates the standard integral:
```
∫_{-1}^{1} f(x) dx
```

Uses extrema points (Chebyshev nodes of the second kind) as sample points.

```python
from chebyshev import ChebyshevQuadrature
import math

q = ChebyshevQuadrature()

# Integrate e^x over [-1, 1]
result = q.clenshaw_curtis_quadrature(lambda x: math.exp(x), n=32)
print(f"∫_{-1}^{1} e^x dx ≈ {result}")
# Exact value is e - 1/e ≈ 2.3504
```

**Parameters:**
- `f` (callable): Function to integrate
- `n` (int): Degree (uses n+1 points, higher = more accurate)

**Returns:** Approximation of ∫_{-1}^{1} f(x) dx

**Note:** Automatically selects O(n²) method for small n and FFT-based O(n log n) for large n (threshold: n ≥ 64).

---

## Convenience Functions

### `get_chebyshev_zeros(n)` — Quick Access to Zeros

```python
from chebyshev import get_chebyshev_zeros

zeros = get_chebyshev_zeros(8)
print(f"Zeros of T_8(x): {zeros}")
```

---

### `get_chebyshev_extrema(n)` — Quick Access to Extrema Points

```python
from chebyshev import get_chebyshev_extrema

extrema = get_chebyshev_extrema(5)
print(f"Extrema of T_5(x): {extrema}")
```

---

### `gauss_chebyshev_integrate(f, n=64)` — Quick Weighted Integration

```python
from chebyshev import gauss_chebyshev_integrate
import math

result = gauss_chebyshev_integrate(lambda x: math.cos(x), n=128)
print(f"Weighted integral ≈ {result}")
```

---

### `clenshaw_curtis_integrate(f, n=32)` — Quick Standard Integration

```python
from chebyshev import clenshaw_curtis_integrate
import math

# Integrate sin(x) over [-1, 1]
result = clenshaw_curtis_integrate(math.sin, n=64)
print(f"∫_{-1}^{1} sin(x) dx ≈ {result}")
# Exact value is 2*sin(1) ≈ 1.683
```

---

## FFT-Based Clenshaw-Curtis (High Performance)

### `clencurt(n)` — Fast Node/Weight Computation

Computes Chebyshev nodes and weights using FFT for large n.

```python
from chebyshev import clencurt
import numpy as np

# Get 1024-point Clenshaw-Curtis quadrature
nodes, weights = clencurt(1023)  # n+1 points
print(f"Number of nodes: {len(nodes)}")
```

---

### `clencurt_quadrature(f, n)` — Fast Integration

```python
from chebyshev import clencurt_quadrature
import math

result = clencurt_quadrature(lambda x: math.exp(-x**2), n=1023)
print(f"∫_{-1}^{1} e^(-x²) dx ≈ {result}")
```

---

## Integration with External Libraries

### SymPy Integration — Symbolic Computation

```python
from chebyshev import generate_sympy_chebyshev, get_sympy_chebyshev_basis
import sympy as sp

# Generate symbolic Chebyshev polynomials up to degree 5
T = generate_sympy_chebyshev(5)
x = sp.symbols('x')

print(f"T_3(x) = {T[3]}")
print(f"T_4(x) = {T[4]}")

# Get symbolic basis for integration
basis = get_sympy_chebyshev_basis(10)
```

---

### NumPy Integration — Array Operations

```python
from chebyshev import generate_numpy_chebyshev, get_numpy_chebyshev_basis
import numpy as np

# Generate NumPy-compatible Chebyshev polynomials
T = generate_numpy_chebyshev(8)
x_vals = np.linspace(-1, 1, 100)

# Evaluate all polynomials at once
results = T(x_vals)  # Returns array of shape (9, 100)
```

---

### mpmath Integration — High Precision

```python
from chebyshev import generate_mpmath_chebyshev, get_mpmath_chebyshev_basis
import mpmath as mp

# Set high precision
mp.mp.dps = 50  # 50 decimal places

# Generate high-precision Chebyshev polynomials
T = generate_mpmath_chebyshev(6)
x_val = mp.mpf('0.123456789')

print(f"T_5({x_val}) = {T[5](x_val)}")
```

---

## Advanced Usage Examples

### Example 1: Polynomial Interpolation at Chebyshev Nodes

```python
from chebyshev import get_chebyshev_zeros, clenshaw_curtis_integrate
import numpy as np

def interpolate_at_chebyshev_nodes(f, n):
    """
    Interpolate function f using Lagrange polynomials at Chebyshev nodes.
    Returns a callable that evaluates the interpolant.
    """
    # Get Chebyshev zeros (optimal interpolation nodes)
    nodes = get_chebyshev_zeros(n)
    values = [f(x) for x in nodes]
    
    def interpolant(x):
        result = 0.0
        for k, x_k in enumerate(nodes):
            L_k = 1.0
            for j, x_j in enumerate(nodes):
                if i != j:
                    L_k *= (x - x_j) / (x_k - x_j)
            result += values[k] * L_k
        return result
    
    return interpolant

# Example: Interpolate sin(x) with 16 nodes
def f(x):
    return np.sin(2*np.pi*x)

interpolant = interpolate_at_chebyshev_nodes(f, 16)
x_test = np.linspace(-1, 1, 100)
errors = [abs(interpolant(x) - f(x)) for x in x_test]
print(f"Max interpolation error: {max(errors):.2e}")
```

### Example 2: Spectral Method for Differential Equations

```python
from chebyshev import ChebyshevQuadrature, get_chebyshev_extrema
import numpy as np

def spectral_derivative(u, n):
    """
    Compute derivative of function given by values u at Chebyshev extrema.
    Uses differentiation matrix approach.
    """
    x = np.array(get_chebyshev_extrema(n))
    D = np.zeros((n+1, n+1))
    
    for i in range(n+1):
        for j in range(n+1):
            if i == j:
                if i == 0 or i == n:
                    D[i, j] = (2*n**2 + 1) / 6
                else:
                    D[i, j] = -x[i]**2 / (2 * (1 - x[i]**2))
            elif i != j:
                D[i, j] = ((-1)**(i+j)) / (x[i] - x[j]) * c_i / c_j
    
    return D @ u

# Solve y'' + y = 0 with y(-1) = 0, y(1) = sin(2)
n = 32
x = np.array(get_chebyshev_extrema(n))
u_exact = np.sin(x)  # Solution: y = sin(x)

# Apply second derivative
D2 = spectral_derivative(u_exact, n) @ spectral_derivative(u_exact, n)
y_approx = -D2
print(f"Max error in solving y'' + y = 0: {np.max(np.abs(y_approx + u_exact)):.2e}")
```

### Example 3: Minimax Approximation (Chebyshev Equioscillation)

```python
from chebyshev import get_chebyshev_extrema, clenshaw_curtis_integrate
import numpy as np

def chebyshev_approximation(f, n):
    """
    Compute Chebyshev series approximation of degree n.
    Uses Clenshaw-Curtis quadrature for coefficient computation.
    """
    coeffs = []
    
    for k in range(n + 1):
        if k == 0:
            # a_0 = (2/π) * ∫_{-1}^{1} f(x)/√(1-x²) dx
            integrand = lambda x: f(x) / np.sqrt(1 - x**2)
            integral = clenshaw_curtis_integrate(integrand, n=2*n + 1)
            a0 = (2/np.pi) * integral
            coeffs.append(a0)
        else:
            # a_k = (2/π) * ∫_{-1}^{1} f(x)*T_k(x)/√(1-x²) dx
            integrand = lambda x: f(x) * np.cos(k * np.arccos(x)) / np.sqrt(1 - x**2)
            integral = clenshaw_curtis_integrate(integrand, n=2*n + 1)
            coeffs.append((2/np.pi) * integral)
    
    return coeffs

def evaluate_chebyshev_series(coeffs, x):
    """
    Evaluate Chebyshev series using Clenshaw's algorithm.
    """
    result = 0.0
    for k, c in enumerate(reversed(coeffs)):
        if k == 0:
            result += c
        elif k == 1:
            result += c * x
        else:
            result = c + 2*x*result - prev_result
        prev_result = result
    return result

# Approximate e^x with degree-8 Chebyshev series
def f(x):
    return np.exp(x)

coeffs = chebyshev_approximation(f, 8)
x_test = np.linspace(-1, 1, 100)
errors = [abs(evaluate_chebyshev_series(coeffs, x) - f(x)) for x in x_test]
print(f"Max approximation error: {max(errors):.2e}")
```

---

## Numerical Stability Notes

### Evaluation Methods

- **Clenshaw's Algorithm** is used internally for evaluating Chebyshev series — it's the most numerically stable method
- For derivatives and integrals, the library uses recurrence relations in the Chebyshev basis (not monomial expansion) to maintain stability

### Quadrature Accuracy

| Method | Weight Function | Best For |
|--------|-----------------|----------|
| Gauss-Chebyshev | 1/√(1-x²) | Functions with endpoint singularities |
| Clenshaw-Curtis | None (standard) | Smooth functions, general purpose |

### Performance Considerations

- **Small n (< 64):** O(n²) direct computation is faster due to lower overhead
- **Large n (≥ 64):** FFT-based method provides O(n log n) complexity
- For very high precision needs (>100 digits), use mpmath integration

---

## API Reference Summary

```python
# Core operations
from chebyshev import (
    chebyshev_derivative_stable,
    chebyshev_integral_stable
)

# Quadrature and nodes
from chebyshev import (
    ChebyshevQuadrature,
    get_chebyshev_zeros,
    get_chebyshev_extrema,
    gauss_chebyshev_integrate,
    clenshaw_curtis_integrate
)

# FFT-based methods
from chebyshev import clencurt, clencurt_quadrature

# External library integrations
from chebyshev import (
    generate_sympy_chebyshev, get_sympy_chebyshev_basis,
    generate_numpy_chebyshev, get_numpy_chebyshev_basis,
    generate_mpmath_chebyshev, get_mpmath_chebyshev_basis
)
```

---

## Mathematical Background

### Definition of Chebyshev Polynomials

Chebyshev polynomials of the first kind Tₙ(x) are defined by:
```
Tₙ(cos θ) = cos(nθ)
```

This leads to the recurrence relation:
```
T₀(x) = 1
T₁(x) = x
T_{n+1}(x) = 2x·Tₙ(x) - T_{n-1}(x)
```

### Key Properties

1. **Orthogonality:** ∫_{-1}^{1} Tₘ(x)Tₙ(x)/√(1-x²) dx = 0 for m ≠ n
2. **Boundedness:** |Tₙ(x)| ≤ 1 for x ∈ [-1, 1]
3. **Equioscillation:** Tₙ(x) oscillates between ±1 exactly n+1 times in [-1, 1]
4. **Minimax Property:** Among all monic polynomials of degree n, (1/2^{n-1})Tₙ(x) has the smallest maximum absolute value on [-1, 1]

---

## License and Attribution

This library is based on mathematical definitions from:
- Abramowitz & Stegun, "Handbook of Mathematical Functions"
- NIST Digital Library of Mathematical Functions (DLMF)
- Trefethen, "Spectral Methods in MATLAB"

The implementation uses numerically stable algorithms adapted from established numerical libraries.
