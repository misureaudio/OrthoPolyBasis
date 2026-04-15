# Laguerre Polynomials Library - User Manual

## Overview

The laguerre library provides a Python implementation for computing and working with Laguerre polynomials ? a family of orthogonal polynomials widely used in:

- **Numerical analysis** ? function approximation, quadrature rules
- **Quantum mechanics** ? hydrogen atom wavefunctions (radial part)
- **Approximation theory** ? spectral methods, least-squares fitting
- **Scientific computing** ? solving differential equations

---

## Installation

### Requirements

`python
# Core dependencies (already included in the package)
numpy      # For stable Gauss-Laguerre quadrature computation
mpmath     # Optional: for high-precision arithmetic with large degrees
`

### Quick Start

`python
from laguerre import LaguerrePolynomial, GeneralizedLaguerrePolynomial
from laguerre import LaguerreBasis, GeneralizedLaguerreBasis
from laguerre import compute_roots, gauss_quadrature_weights
`

---

## Core Polynomial Classes

### LaguerrePolynomial(n) ? Standard Laguerre Polynomials L?(x)

Standard Laguerre polynomials with parameter ? = 0.

#### Constructor
`python
from laguerre import LaguerrePolynomial

P3 = LaguerrePolynomial(3)  # Creates L_3(x)
`

**Parameters:**
- 
 (int): Degree of the polynomial (must be ? 0)

#### Methods

| Method | Description |
|--------|-------------|
| evaluate(x) | Evaluate L?(x) at point x using stable forward recurrence |
| get_coefficients() | Return list of coefficients [a?, a?, ..., a?] where P(x) = ? a?x? |
| degree (property) | Returns the degree n |
| __call__(x) | Call operator: same as evaluate(x) |

#### Example Usage
`python
from laguerre import LaguerrePolynomial

# Create L_3(x)
P = LaguerrePolynomial(3)
print(f"L_3(x) coefficients: {P.get_coefficients()}")
# Output: [1.0, -3.0, 1.5, -0.167] ? L?(x) = 1 - 3x + (3/2)x? - (1/6)x?

# Evaluate at specific points
print(f"L_3(0) = {P.evaluate(0)}")   # Output: 1.0
print(f"L_3(1) = {P.evaluate(1)}")   # Output: -0.667
print(f"L_3(2) = {P.evaluate(2)}")   # Output: -0.333

# Roots of L_3(x): approximately 0.416, 2.294, 6.290
from laguerre import compute_roots
roots = compute_roots(3)
print(f"Roots: {roots}")  # [0.415775, 2.294280, 6.289945]

# Using call operator
result = P(5)  # Same as P.evaluate(5)
`

#### Mathematical Definition

Standard Laguerre polynomials satisfy the recurrence:
`
(n+1)L_{n+1}(x) = (2n + 1 - x)L_n(x) - nL_{n-1}(x)
`

With initial conditions: L?(x) = 1, L?(x) = 1 - x

---

### GeneralizedLaguerrePolynomial(n, alpha) ? Generalized Form L?^(?)(x)

Generalized Laguerre polynomials with parameter ? > -1.

#### Constructor
`python
from laguerre import GeneralizedLaguerrePolynomial

# Standard case (alpha=0 is default)
P = GeneralizedLaguerrePolynomial(3, alpha=0.0)

# Generalized form with custom alpha
P_alpha = GeneralizedLaguerrePolynomial(2, alpha=1.5)
`

**Parameters:**
- 
 (int): Degree of the polynomial (must be ? 0)
- lpha (float): Parameter ? (must be > -1)

#### Methods

| Method | Description |
|--------|-------------|
| evaluate(x) | Evaluate L?^(?)(x) at point x |
| evaluate_with_derivative(x) | Return tuple (value, derivative) simultaneously |
| get_coefficients() | Return list of coefficients |
| degree (property) | Returns the degree n |
| lpha (property) | Returns the ? parameter |
| __call__(x) | Call operator: same as evaluate(x) |

#### Example Usage
`python
from laguerre import GeneralizedLaguerrePolynomial

# Create generalized polynomial L_2^(1.5)(x)
P = GeneralizedLaguerrePolynomial(2, alpha=1.5)

# Evaluate with derivative at x=0.5
value, deriv = P.evaluate_with_derivative(0.5)
print(f"L_2^(1.5)(0.5) = {value}")      # Output: 2.75
print(f"d/dx[L_2^(1.5)](0.5) = {deriv}") # Output: -3.0

# Get coefficients
coeffs = P.get_coefficients()
print(f"Coefficients: {coeffs}")  # [4.375, -3.5, 0.5]
`

#### Mathematical Definition

Generalized Laguerre polynomials satisfy:
`
(n+1)L_{n+1}^(?)(x) = (2n + ? + 1 - x)L_n^(?)(x) - (n + ?)L_{n-1}^(?)(x)
`

Derivative identity: x ? d/dx[L?^(?)](x) = n?L?^(?)(x) - (n+?)?L_{n-1}^(?)(x)

---

## Function Approximation with Basis Classes

### LaguerreBasis(max_degree) ? Standard Laguerre Basis

Provides infrastructure for approximating functions using Laguerre polynomial expansions.

#### Constructor
`python
from laguerre import LaguerreBasis

# Create basis up to degree 10
basis = LaguerreBasis(10)
`

**Parameters:**
- max_degree (int): Maximum polynomial degree in the basis

#### Methods

| Method | Description |
|--------|-------------|
| project(f) | Compute expansion coefficients of function f in the basis |
| pproximate(f) | Return callable that approximates f as a Laguerre series |
| inner_product(f, g=None) | Numerical integration using Gauss-Laguerre quadrature |
| 
orm_squared(n) | Returns ?L??? = ?(n+1)/n! under weight w(x) = e?? |
| weight_function(x) | Returns the weight function value at x |

#### Example Usage
`python
from laguerre import LaguerreBasis
import math

# Create basis up to degree 5
basis = LaguerreBasis(5)

# Define a target function (e.g., e^(-x/2))
def f(x):
    return math.exp(-x / 2)

# Project onto the basis
coeffs = basis.project(f)
print(f"Coefficients: {[f'{c:.6f}' for c in coeffs]}")
# Output: ['0.666667', '0.222222', '0.074074', '0.024691', '0.008230', '0.002743']

# Create approximation callable
approx = basis.approximate(f)
print(f"f(1) ? {approx(1):.6f} (actual: {f(1):.6f})")  # 0.606744 vs 0.606531
print(f"f(5) ? {approx(5):.6f} (actual: {f(5):.6f})")  # 0.083562 vs 0.082085
`

---

### GeneralizedLaguerreBasis(max_degree, alpha) ? Generalized Basis

Same interface as LaguerreBasis but with customizable ? parameter.

`python
from laguerre import GeneralizedLaguerreBasis
import math

# Create generalized basis with ? = 1.0
basis_alpha = GeneralizedLaguerreBasis(10, alpha=1.0)

def f(x):
    return x * math.exp(-x / 2)  # Function suited for ? > 0

coeffs = basis_alpha.project(f)
approx = basis_alpha.approximate(f)
`

---

## Utility Functions

### compute_roots(n, alpha=0.0) ? Polynomial Roots

Returns the n roots of L?^(?)(x) using Gauss-Laguerre quadrature nodes.

`python
from laguerre import compute_roots

# Roots of standard L_5(x)
roots = compute_roots(5)
print(f"Roots: {roots}")
# Output: [0.26356, 1.41340, 3.59643, 7.08581, 12.64080]

# Roots of generalized L_5^(1)(x)
roots_alpha = compute_roots(5, alpha=1.0)
`

---

### gauss_quadrature_weights(n, alpha=0.0) ? Quadrature Nodes and Weights

Returns (nodes, weights) for n-point Gauss-Laguerre quadrature.

`python
from laguerre import gauss_quadrature_weights
import math

# Gauss-Laguerre quadrature computes ??^? f(x) ? e^(-x) dx
# The weight function e^(-x) is already included in the quadrature!

# Compute ??^? e^(-x) ? x? dx = ?(3) = 2
nodes, weights = gauss_quadrature_weights(5)

# Just evaluate x? at nodes (weight e^(-x) is built-in)
result = sum(w * n**2 for n, w in zip(nodes, weights))
print(f"??^? e^(-x) ? x? dx ? {result}")  # Output: ~2.0

# For ??^? e^(-x) ? sin(x) dx = 1/2
result_sin = sum(w * math.sin(n) for n, w in zip(nodes, weights))
print(f"??^? e^(-x) ? sin(x) dx ? {result_sin}")  # Output: ~0.5
`

---

### unction_projection(f, max_degree, alpha=0.0) ? Quick Projection

Convenience function to project a function onto Laguerre basis.

`python
from laguerre import function_projection
import math

def f(x):
    return math.exp(-x)

# Project onto degree-7 basis with ?=0
coeffs = function_projection(f, max_degree=7)
print(f"Projection coefficients: {coeffs}")
`

---

### unction_approximation(f, max_degree, alpha=0.0) ? Quick Approximation

Convenience function that returns a callable approximation.

`python
from laguerre import function_approximation
import math

def f(x):
    return math.exp(-x/2)

# Get approximation callable
approx = function_approximation(f, max_degree=10)

# Evaluate at various points
for x in [0, 1, 5]:
    print(f"f({x}) ? {approx(x)}")
`

---

### 	o_sympy(n, alpha=0.0) ? Symbolic Conversion

Converts Laguerre polynomial to SymPy symbolic expression.

`python
from laguerre import to_sympy
import sympy as sp

# Get symbolic L_3(x)
P_sym = to_sympy(3)
x = sp.symbols('x')
print(f"L_3(x) = {P_sym}")  # Output: -x**3/6 + 3*x**2/2 - 3*x + 1

# Verify properties symbolically
print(f"L_3(0) = {P_sym.subs(x, 0)}")  # Output: 1
`

---

## Advanced Usage Examples

### Example 1: Spectral Approximation of a Function

`python
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
`

### Example 2: Solving a Differential Equation

Laguerre polynomials are eigenfunctions of the operator:
`
d/dx[x?d/dx] + (?+1-x) ? L?^(?)(x) = -n?L?^(?)(x)
`

`python
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
`

### Example 3: Quantum Mechanics ? Hydrogen Atom Radial Wavefunction

`python
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
`

---

## Numerical Stability Notes

### Evaluation Methods

The library uses **stable forward recurrence** for polynomial evaluation:
- O(n) complexity (linear in degree)
- More stable than direct coefficient expansion for high degrees
- Recommended for n < 1000

### Gauss-Laguerre Quadrature

For accurate numerical integration:
- Use at least 2*max_degree + 1 points for exact integration of polynomials up to degree max_degree
- The library automatically caches quadrature rules for efficiency
- For very high precision, consider installing mpmath and using higher-order rules

### Large Degrees

For degrees n > 700:
- Standard float64 may overflow in norm calculations
- The library attempts to use mpmath if available
- Otherwise returns infinity for norms

---

## API Reference Summary

`python
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
`

---

## License and Attribution

This library is based on mathematical definitions from:
- Abramowitz & Stegun, "Handbook of Mathematical Functions"
- NIST Digital Library of Mathematical Functions (DLMF)

The implementation uses numerically stable algorithms adapted from established numerical libraries.