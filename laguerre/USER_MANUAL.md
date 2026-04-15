# Laguerre Polynomials Library - User Manual

## Overview

The `laguerre` library provides a Python implementation for computing and working with Laguerre polynomials — a family of orthogonal polynomials widely used in:

- **Numerical analysis** — function approximation, quadrature rules
- **Quantum mechanics** — hydrogen atom wavefunctions (radial part)
- **Approximation theory** — spectral methods, least-squares fitting
- **Scientific computing** — solving differential equations

---

## Installation

### Requirements

```python
# Core dependencies (already included in the package)
numpy      # For stable Gauss-Laguerre quadrature computation
mpmath     # Optional: for high-precision arithmetic with large degrees
```

### Quick Start

```python
from laguerre import LaguerrePolynomial, GeneralizedLaguerrePolynomial
from laguerre import LaguerreBasis, GeneralizedLaguerreBasis
from laguerre import compute_roots, gauss_quadrature_weights
```

---

## Core Polynomial Classes

### `LaguerrePolynomial(n)` — Standard Laguerre Polynomials Lₙ(x)

Standard Laguerre polynomials with parameter α = 0.

#### Constructor
```python
from laguerre import LaguerrePolynomial

P3 = LaguerrePolynomial(3)  # Creates L_3(x)
```

**Parameters:**
- `n` (int): Degree of the polynomial (must be ≥ 0)

#### Methods

| Method | Description |
|--------|-------------|
| `evaluate(x)` | Evaluate Lₙ(x) at point x using stable forward recurrence |
| `get_coefficients()` | Return list of coefficients [a₀, a₁, ..., aₙ] where P(x) = Σ aᵢxⁱ |
| `degree` (property) | Returns the degree n |
| `__call__(x)` | Call operator: same as evaluate(x) |

#### Example Usage
```python
from laguerre import LaguerrePolynomial

# Create L_3(x)
P = LaguerrePolynomial(3)
print(f"L_3(x) coefficients: {P.get_coefficients()}")
# Output: [1.0, -3.0, 1.5, -0.167] → L₃(x) = 1 - 3x + (3/2)x² - (1/6)x³

# Evaluate at specific points
print(f"L_3(0) = {P.evaluate(0)}")   # Output: 1.0
print(f"L_3(1) = {P.evaluate(1)}")   # Output: -0.667
print(f"L_3(2) = {P.evaluate(2)}")   # Output: -0.333

# Using call operator
result = P(5)  # Same as P.evaluate(5)
```

#### Mathematical Definition

Standard Laguerre polynomials satisfy the recurrence:
```
(n+1)L_{n+1}(x) = (2n + 1 - x)L_n(x) - nL_{n-1}(x)
```

With initial conditions: L₀(x) = 1, L₁(x) = 1 - x

---

### `GeneralizedLaguerrePolynomial(n, alpha)` — Generalized Form Lₙ^(α)(x)

Generalized Laguerre polynomials with parameter α > -1.

#### Constructor
```python
from laguerre import GeneralizedLaguerrePolynomial

# Standard case (alpha=0 is default)
P = GeneralizedLaguerrePolynomial(3, alpha=0.0)

# Generalized form with custom alpha
P_alpha = GeneralizedLaguerrePolynomial(2, alpha=1.5)
```

**Parameters:**
- `n` (int): Degree of the polynomial (must be ≥ 0)
- `alpha` (float): Parameter α (must be > -1)

#### Methods

| Method | Description |
|--------|-------------|
| `evaluate(x)` | Evaluate Lₙ^(α)(x) at point x |
| `evaluate_with_derivative(x)` | Return tuple (value, derivative) simultaneously |
| `get_coefficients()` | Return list of coefficients |
| `degree` (property) | Returns the degree n |
| `alpha` (property) | Returns the α parameter |
| `__call__(x)` | Call operator: same as evaluate(x) |

#### Example Usage
```python
from laguerre import GeneralizedLaguerrePolynomial

# Create generalized polynomial L_2^(1.5)(x)
P = GeneralizedLaguerrePolynomial(2, alpha=1.5)

# Evaluate with derivative at x=0.5
value, deriv = P.evaluate_with_derivative(0.5)
print(f"L_2^(1.5)(0.5) = {value}")
print(f"d/dx[L_2^(1.5)](0.5) = {deriv}")

# Get coefficients
coeffs = P.get_coefficients()
print(f"Coefficients: {coeffs}")
```

#### Mathematical Definition

Generalized Laguerre polynomials satisfy:
```
(n+1)L_{n+1}^(α)(x) = (2n + α + 1 - x)L_n^(α)(x) - (n + α)L_{n-1}^(α)(x)
```

Derivative identity: `x · d/dx[Lₙ^(α)](x) = n·Lₙ^(α)(x) - (n+α)·L_{n-1}^(α)(x)`

---

## Function Approximation with Basis Classes

### `LaguerreBasis(max_degree)` — Standard Laguerre Basis

Provides infrastructure for approximating functions using Laguerre polynomial expansions.

#### Constructor
```python
from laguerre import LaguerreBasis

# Create basis up to degree 10
basis = LaguerreBasis(10)
```

**Parameters:**
- `max_degree` (int): Maximum polynomial degree in the basis

#### Methods

| Method | Description |
|--------|-------------|
| `project(f)` | Compute expansion coefficients of function f in the basis |
| `approximate(f)` | Return callable that approximates f as a Laguerre series |
| `inner_product(f, g=None)` | Numerical integration using Gauss-Laguerre quadrature |
| `norm_squared(n)` | Returns ‖Lₙ‖² = Γ(n+1)/n! under weight w(x) = e⁻ˣ |
| `weight_function(x)` | Returns the weight function value at x |

#### Example Usage
```python
from laguerre import LaguerreBasis
import math

# Create basis up to degree 5
basis = LaguerreBasis(5)

# Define a target function (e.g., e^(-x/2))
def target_function(x):
    return math.exp(-0.5 * x)

# Project: get expansion coefficients
coeffs = basis.project(target_function)
print(f"Expansion coefficients: {coeffs}")

# Approximate: get a callable approximation
approx = basis.approximate(target_function)

# Evaluate the approximation at various points
for x in [0, 1, 5, 10]:
    exact = target_function(x)
    approx_val = approx(x)
    print(f"x={x}: exact={exact:.6f}, approx={approx_val:.6f}")

# Compute inner product with itself (should equal norm squared sum)
ip = basis.inner_product(target_function)
print(f"Inner product: {ip}")
```

---

### `GeneralizedLaguerreBasis(max_degree, alpha)` — Generalized Basis

Same interface as LaguerreBasis but with weight w(x) = x^α · e⁻ˣ.

#### Constructor
```python
from laguerre import GeneralizedLaguerreBasis

# Standard case (alpha=0)
basis = GeneralizedLaguerreBasis(10, alpha=0.0)

# Generalized form
basis_alpha = GeneralizedLaguerreBasis(8, alpha=2.5)
```

#### Example Usage
```python
from laguerre import GeneralizedLaguerreBasis
import math

# Create basis with α = 1 (weight: x·e⁻ˣ)
basis = GeneralizedLaguerreBasis(6, alpha=1.0)

def f(x):
    return math.exp(-x) * math.sin(x)

# Project and approximate
coeffs = basis.project(f)
approx = basis.approximate(f)

print(f"Approximation coefficients: {coeffs}")
```

---

## Utility Functions

### `compute_roots(n, alpha=0.0)` — Compute Polynomial Roots

Returns the n roots of Lₙ^(α)(x) using stable numerical methods.

```python
from laguerre import compute_roots

# Get 5 roots of standard Laguerre polynomial
roots = compute_roots(5)
print(f"Roots of L_5(x): {roots}")

# Get roots for generalized case
roots_alpha = compute_roots(4, alpha=1.5)
```

---

### `gauss_quadrature_weights(n, alpha=0.0)` — Gauss-Laguerre Quadrature

Returns nodes and weights for n-point Gauss-Laguerre quadrature.

```python
from laguerre import gauss_quadrature_weights

# Get 10-point quadrature rule
nodes, weights = gauss_quadrature_weights(10)
print(f"Nodes: {nodes}")
print(f"Weights: {weights}")

# Use for numerical integration:
# ∫₀^∞ e⁻ˣ f(x) dx ≈ Σᵢ wᵢ · f(nodeᵢ)
def integrand(x):
    return math.exp(-x) * x**2  # Should integrate to Γ(3) = 2

result = sum(w * integrand(n) for n, w in zip(nodes, weights))
print(f"Integral ≈ {result}")  # Should be close to 2.0
```

---

### `function_projection(f, max_degree, alpha=0.0)` — Quick Projection

Convenience function to project a function onto Laguerre basis.

```python
from laguerre import function_projection
import math

def f(x):
    return math.exp(-x)

# Project onto degree-7 basis with α=0
coeffs = function_projection(f, max_degree=7)
print(f"Projection coefficients: {coeffs}")
```

---

### `function_approximation(f, max_degree, alpha=0.0)` — Quick Approximation

Convenience function that returns a callable approximation.

```python
from laguerre import function_approximation
import math

def f(x):
    return math.exp(-x/2)

# Get approximation callable
approx = function_approximation(f, max_degree=10)

# Evaluate at various points
for x in [0, 1, 5]:
    print(f"f({x}) ≈ {approx(x)}")
```

---

### `to_sympy(n, alpha=0.0)` — Symbolic Conversion

Converts Laguerre polynomial to SymPy symbolic expression.

```python
from laguerre import to_sympy
import sympy as sp

# Get symbolic L_3(x)
P_sym = to_sympy(3)
x = sp.symbols('x')
print(f"L_3(x) = {P_sym}")  # Should show: -x**3 + 6*x**2 - 15*x + 10

# Verify properties symbolically
print(f"L_3(0) = {P_sym.subs(x, 0)}")
```

---

## Advanced Usage Examples

### Example 1: Spectral Approximation of a Function

```python
from laguerre import LaguerreBasis
import numpy as np
import math

# Target function: e^(-x/2)
def target(x):
    return math.exp(-0.5 * x)

# Create high-order basis
basis = LaguerreBasis(20)

# Get approximation
approx = basis.approximate(target)

# Compute error over a range
xs = np.linspace(0, 20, 100)
errors = [abs(approx(x) - target(x)) for x in xs]
print(f"Max absolute error: {max(errors):.2e}")
```

### Example 2: Solving a Differential Equation

Laguerre polynomials are eigenfunctions of the operator:
```
d/dx[x·d/dx] + (α+1-x) → Lₙ^(α)(x) = -n·Lₙ^(α)(x)
```

```python
from laguerre import GeneralizedLaguerrePolynomial
import numpy as np

def verify_eigenvalue(n, alpha=0):
    P = GeneralizedLaguerrePolynomial(n, alpha)
    xs = np.linspace(0.1, 10, 50)  # Avoid x=0 for derivative check
    
    errors = []
    for x in xs:
        val, deriv = P.evaluate_with_derivative(x)
        d2_val = (deriv - val / x) / x if abs(x) > 1e-10 else 0
        lhs = x * d2_val + deriv + (alpha + 1 - x) * deriv
        rhs = -n * val
        errors.append(abs(lhs - rhs))
    
    return max(errors)

print(f"Eigenvalue verification error: {verify_eigenvalue(5):.2e}")
```

### Example 3: Quantum Mechanics — Hydrogen Atom Radial Wavefunction

```python
from laguerre import GeneralizedLaguerrePolynomial, compute_roots
import numpy as np
import math

def hydrogen_radial(r, n, l):
    """
    Radial wavefunction for hydrogen atom.
    Uses generalized Laguerre polynomials L_{n-l-1}^{2l+1}(2r/n)
    """
    rho = 2 * r / n
    max_n = n - l - 1
    alpha = 2 * l + 1
    
    # Get normalization constant (simplified)
    norm = math.sqrt((math.factorial(max_n) / (2*n*math.factorial(max_n+alpha))))
    
    # Laguerre polynomial part
    L = GeneralizedLaguerrePolynomial(max_n, alpha)
    laguerre_val = L.evaluate(rho)
    
    return norm * math.exp(-rho/2) * (rho**l) * laguerre_val

# Plot 1s orbital (n=1, l=0)
r_values = np.linspace(0, 10, 100)
psi_1s = [hydrogen_radial(r, 1, 0) for r in r_values]
```

---

## Numerical Stability Notes

### Evaluation Methods

The library uses **stable forward recurrence** for polynomial evaluation:
- O(n) complexity (linear in degree)
- More stable than direct coefficient expansion for high degrees
- Recommended for n < 1000

### Gauss-Laguerre Quadrature

For accurate numerical integration:
- Use at least `2*max_degree + 1` points for exact integration of polynomials up to degree `max_degree`
- The library automatically caches quadrature rules for efficiency
- For very high precision, consider installing `mpmath` and using higher-order rules

### Large Degrees

For degrees n > 700:
- Standard float64 may overflow in norm calculations
- The library attempts to use `mpmath` if available
- Otherwise returns infinity for norms

---

## API Reference Summary

```python
# Polynomial classes
from laguerre import LaguerrePolynomial, GeneralizedLaguerrePolynomial

# Basis classes  
from laguerre import LaguerreBasis, GeneralizedLaguerreBasis

# Utility functions
from laguerre import (
    compute_roots,
    gauss_quadrature_weights,
    function_projection,
    function_approximation,
    to_sympy
)
```

---

## License and Attribution

This library is based on mathematical definitions from:
- Abramowitz & Stegun, "Handbook of Mathematical Functions"
- NIST Digital Library of Mathematical Functions (DLMF)

The implementation uses numerically stable algorithms adapted from established numerical libraries.
