# Hermite Polynomials Library - User Manual

## Overview

The `hermite` library provides a Python implementation for computing and working with Hermite polynomials — a family of orthogonal polynomials widely used in:

- **Quantum mechanics** — harmonic oscillator wavefunctions, quantum field theory
- **Probability theory** — Edgeworth expansions, statistical distributions
- **Numerical analysis** — Gaussian quadrature (Gauss-Hermite), spectral methods
- **Approximation theory** — function approximation on the real line with weight e^{-x^2}

---

## Architecture: Three-Layer Stack

The library uses a layered architecture for optimal performance and accuracy:

| Layer | Module | Purpose |
|-------|--------|----------|
| **Source** | `symbolic.py` | Exact rational coefficients via SymPy — ground truth |
| **Bridge** | `high_precision.py` | 50+ decimal precision via mpmath — critical for roots/weights when n > 100 |
| **Engine** | `numerical.py` | Fast NumPy/SciPy array operations — optimized for bulk data processing |
| **Client** | `integration.py` | Quadrature and projection utilities — orchestrates the three layers |

---

## Installation

### Requirements

```python
# Core dependencies (already included in the package)
numpy      # For numerical evaluation and array operations
scipy      # Optional: for additional numerical methods
sympy      # Required: for symbolic computation
mpmath     # Required: for high-precision arithmetic
```

### Quick Start

```python
from hermite import (
    HermiteSymbolic,
    HermiteMPMath,
    HermitePolynomial,
    GaussHermiteQuadrature,
    HermiteProjection,
)
```

---

## Core Polynomial Classes

### `HermiteSymbolic` — Exact Symbolic Representation

Provides exact rational coefficients using SymPy's arbitrary-precision arithmetic.

#### Constructor
```python
from hermite import HermiteSymbolic

# Create H_5(x) (physicist's convention)
P = HermiteSymbolic(5, convention="physicist")
print(P)  # Output: 32*x**5 - 160*x**3 + 120*x
```

**Parameters:**
- `degree` (int): Degree of the polynomial (must be ≥ 0)
- `convention` (str): Either "physicist" (H_n) or "probabilist" (He_n)
- `x_symbol`: Optional custom SymPy symbol for x

#### Methods

| Method | Description |
|--------|-------------|
| `derivative(order=1)` | Compute derivative using Hermite recurrence relations |
| `indefinite_integral()` | Compute indefinite integral |
| `get_coefficients(ascending=False)` | Return list of rational coefficients |
| `to_numerical_coeffs()` | Export as Python floats for numerical.py |
| `evaluate_symbolic(point)` | Evaluate at a symbolic point |

#### Example Usage
```python
from hermite import HermiteSymbolic
import sympy as sp

# Create H_4(x)
P = HermiteSymbolic(4, convention="physicist")
print(f"H_4(x) = {P}")  # 16*x**4 - 48*x**2 + 12

# Get coefficients
coeffs = P.get_coefficients()
print(f"Coefficients: {coeffs}")  # [16, 0, -48, 0, 12]

# Derivative
P_deriv = P.derivative(2)
print(f"H_4''(x) = {P_deriv}")  # 192*x**2 - 96

# Evaluate symbolically
result = P.evaluate_symbolic(sp.sqrt(2))
print(f"H_4(sqrt(2)) = {result}")
```

---

### `HermiteMPMath` — High-Precision Representation

Provides high-precision (50+ decimal places) evaluation using mpmath.

#### Constructor
```python
from hermite import HermiteMPMath

# Create H_3(x) with 100 decimal precision
P = HermiteMPMath(3, convention="physicist", dps=100)
```

**Parameters:**
- `degree` (int): Degree of the polynomial
- `convention` (str): Either "physicist" or "probabilist"
- `coeffs`: Optional list of coefficients for custom construction
- `dps` (int): Decimal precision (default: 50)

#### Methods

| Method | Description |
|--------|-------------|
| `evaluate(x)` | Evaluate at point x with high precision |
| `derivative(order=1)` | Compute derivative |
| `get_coefficients(ascending=False)` | Return list of mpmath.mpf coefficients |
| `to_numerical_coeffs()` | Export as Python floats |

#### Example Usage
```python
from hermite import HermiteMPMath
import mpmath as mp

# Set high precision
mp.mp.dps = 100

P = HermiteMPMath(5, convention="physicist", dps=100)
x_val = mp.mpf('0.123456789')
result = P.evaluate(x_val)
print(f"H_5({x_val}) = {result}")  # High-precision result
```

---

### `HermitePolynomial` — Fast Numerical Evaluation

Provides fast numerical evaluation using NumPy for bulk data processing.

#### Constructor
```python
from hermite import HermitePolynomial
import numpy as np

# Create H_4(x) (physicist's convention)
P = HermitePolynomial(4, convention="physicist")
```

**Parameters:**
- `degree` (int): Degree of the polynomial
- `convention` (str): Either "physicist" or "probabilist"
- `coeffs`: Optional array of coefficients for custom construction

#### Methods

| Method | Description |
|--------|-------------|
| `evaluate(x)` | Evaluate at point(s) x using NumPy polyval |
| `derivative(order=1)` | Compute derivative as new HermitePolynomial |
| `integrate()` | Compute indefinite integral |
| `get_coefficients(ascending=False)` | Return coefficient array |
| `verify_against_scipy(test_points=None)` | Verify against SciPy implementation |

#### Example Usage
```python
from hermite import HermitePolynomial
import numpy as np

# Create H_3(x)
P = HermitePolynomial(3, convention="physicist")
print(f"H_3(x) coefficients: {P.get_coefficients()}")  # [8.0, 0.0, -12.0, 0.0]

# Evaluate at single point
result = P.evaluate(2.0)
print(f"H_3(2) = {result}")  # 40.0

# Evaluate at multiple points (vectorized)
x_vals = np.array([-1, -0.5, 0, 0.5, 1])
results = P.evaluate(x_vals)
print(f"H_3 evaluated: {results}")  # [-8., 4., 0., -4., 8.]

# Derivative
P_deriv = P.derivative(2)
print(f"H_3''(x) coefficients: {P_deriv.get_coefficients()}")  # [16.0, 0.0]
```

---

## Quadrature and Integration

### `GaussHermiteQuadrature` — Gauss-Hermite Quadrature

Computes quadrature points (roots of H_n) and weights for integrals with weight e^{-x^2}.

#### Constructor
```python
from hermite import GaussHermiteQuadrature

# Create 32-point quadrature rule
quad = GaussHermiteQuadrature(32, dps=50)
```

**Parameters:**
- `n` (int): Number of quadrature points
- `dps` (int): Decimal precision for root/weight computation (default: 50)

#### Properties

| Property | Description |
|----------|-------------|
| `roots` | Array of n quadrature points (roots of H_n) |
| `weights` | Array of n quadrature weights |

#### Methods

| Method | Description |
|--------|-------------|
| `integrate(f, vectorized=True)` | Compute ∫_{-∞}^{+∞} f(x)e^{-x^2} dx |
| `integrate_with_weight(f, custom_weight=None)` | Add custom weight function |

#### Example Usage
```python
from hermite import GaussHermiteQuadrature
import numpy as np

# Create 64-point quadrature rule
quad = GaussHermiteQuadrature(64)

print(f"Number of points: {len(quad.roots)}")
print(f"First few roots: {quad.roots[:5]}")
print(f"Sum of weights: {np.sum(quad.weights):.10f}")  # Should be √π ≈ 1.77245

# Integrate f(x) = e^{-x^2} * x^2
# Result should be ∫_{-∞}^{+∞} x^2 e^{-x^2} dx = √π/2
def integrand(x):
    return x**2

result = quad.integrate(integrand)
print(f"∫ x²e^{-x²}dx ≈ {result}")  # Should be close to 0.88623 (√π/2)

# Integrate with custom weight: ∫ f(x) e^{-x^2} w(x) dx
def weighted_integrand(x):
    return np.cos(x)

custom_weight = lambda x: np.exp(-x**4)  # Additional weight
result_weighted = quad.integrate_with_weight(weighted_integrand, custom_weight)
print(f"Weighted integral ≈ {result_weighted}")
```

---

### `HermiteProjection` — Function Projection onto Hermite Basis

Projects functions onto the Hermite polynomial basis to compute expansion coefficients.

#### Constructor
```python
from hermite import HermiteProjection

# Create projector for degree up to 20
projector = HermiteProjection(20, quadrature_points=50)
```

**Parameters:**
- `n_max` (int): Maximum degree of basis polynomials
- `quadrature_points`: Number of quadrature points (default: 2*n_max + 1)
- `dps`: Precision for computation (default: 50)

#### Methods

| Method | Description |
|--------|-------------|
| `project(f)` | Compute Hermite coefficients [c_0, c_1, ..., c_n] |
| `reconstruct(coeffs, x)` | Reconstruct function from coefficients at point(s) x |
| `project_and_reconstruct(f, x)` | Project and reconstruct in one call |

#### Example Usage
```python
from hermite import HermiteProjection
import numpy as np

# Create projector for degree up to 15
projector = HermiteProjection(15)

# Define a function to project: f(x) = e^{-x^2/4}
def target_function(x):
    return np.exp(-x**2 / 4)

# Project onto Hermite basis
coeffs = projector.project(target_function)
print(f"Number of coefficients: {len(coeffs)}")
print(f"First 5 coefficients: {coeffs[:5]}")

# Reconstruct at specific points
x_test = np.linspace(-3, 3, 100)
reconstructed = projector.reconstruct(coeffs, x_test)
original = target_function(x_test)

# Compute error
error = np.max(np.abs(reconstructed - original))
print(f"Max reconstruction error: {error:.2e}")
```

---

## Utility Functions

### `hermite_symbolic_basis(n_max, convention="physicist")` — Symbolic Basis Generator

```python
from hermite import hermite_symbolic_basis

# Generate basis {H_0, H_1, ..., H_n} symbolically
basis = hermite_symbolic_basis(5)
for i, P in enumerate(basis):
    print(f"H_{i}(x) = {P}")
```

---

### `hermite_high_precision_basis(n_max, convention="physicist", dps=50)` — High-Precision Basis Generator

```python
from hermite import hermite_high_precision_basis
import mpmath as mp

mp.mp.dps = 100
basis = hermite_high_precision_basis(3, dps=100)
print(f"H_2(x) coefficients: {basis[2].get_coefficients()}")
```

---

### `hermite_numerical_basis(n_max, convention="physicist")` — Numerical Basis Generator

```python
from hermite import hermite_numerical_basis
import numpy as np

basis = hermite_numerical_basis(4)
x_vals = np.linspace(-2, 2, 100)
for i, P in enumerate(basis):
    y_vals = P.evaluate(x_vals)
    print(f"H_{i}(x) evaluated at {len(y_vals)} points")
```

---

### `hermite_transform(f, n_max, quadrature_points=None, dps=50)` — Quick Transform

Convenience function for single-shot Hermite transform.

```python
from hermite import hermite_transform
import numpy as np

def f(x):
    return np.exp(-x**2 / 4)

coeffs, quad = hermite_transform(f, n_max=10)
print(f"Hermite coefficients: {coeffs}")
```

---

### `inverse_hermite_transform(coeffs, x)` — Quick Reconstruction

Convenience function for reconstructing from Hermite coefficients.

```python
from hermite import inverse_hermite_transform
import numpy as np

# Some Hermite coefficients
coeffs = np.array([1.0, 0.0, -0.5, 0.0, 0.25])
x_vals = np.linspace(-2, 2, 10)
reconstructed = inverse_hermite_transform(coeffs, x_vals)
print(f"Reconstructed values: {reconstructed}")
```

---

### `verify_quadrature_accuracy(n, dps=50)` — Quadrature Verification

Verifies quadrature accuracy on known integrals.

```python
from hermite import verify_quadrature_accuracy

results = verify_quadrature_accuracy(32)
print(f"Max norm error: {results['max_norm_error']:.2e}")
print(f"Max orthogonality error: {results['max_ortho_error']:.2e}")
```

---

## Advanced Usage Examples

### Example 1: Quantum Harmonic Oscillator Wavefunctions

```python
from hermite import HermitePolynomial, GaussHermiteQuadrature
import numpy as np
import math

def harmonic_oscillator_wavefunction(n, x, omega=1.0, m=1.0):
    """
    Compute the n-th energy eigenstate of quantum harmonic oscillator.
    
    ψ_n(x) = (mω/(πℏ))^(1/4) * 1/sqrt(2^n * n!) * H_n(ξ) * exp(-ξ^2/2)
    where ξ = sqrt(mω/ℏ) * x
    """
    # Natural units: ℏ = m = ω = 1
    xi = np.sqrt(x)
    
    # Get Hermite polynomial H_n(ξ)
    H_n = HermitePolynomial(n, convention="physicist")
    H_n_val = H_n.evaluate(xi)
    
    # Normalization constant
    norm = (1 / math.pi)**0.25 / np.sqrt(2**n * math.factorial(n))
    
    return norm * H_n_val * np.exp(-xi**2 / 2)

# Plot ground state and first excited state
x_vals = np.linspace(-4, 4, 1000)
psi_0 = harmonic_oscillator_wavefunction(0, x_vals)
psi_1 = harmonic_oscillator_wavefunction(1, x_vals)

print(f"∫ |ψ_0|^2 dx = {np.trapz(psi_0**2, x_vals):.6f}")  # Should be ~1.0
print(f"∫ |ψ_1|^2 dx = {np.trapz(psi_1**2, x_vals):.6f}")  # Should be ~1.0
```

### Example 2: Edgeworth Expansion for Probability Distributions

```python
from hermite import HermiteProjection, GaussHermiteQuadrature
import numpy as np

def edgeworth_expansion(pdf_samples, n_terms=5):
    """
    Compute Edgeworth expansion coefficients from samples.
    Uses Hermite polynomials to approximate deviation from Gaussian.
    """
    # Standardize samples
    mean = np.mean(pdf_samples)
    std = np.std(pdf_samples)
    standardized = (pdf_samples - mean) / std
    
    # Compute moments
    m2 = np.mean(standardized**2)  # Should be ~1
    m3 = np.mean(standardized**3)  # Skewness
    m4 = np.mean(standardized**4)  # Kurtosis (excess: m4 - 3)
    
    # Edgeworth expansion coefficients
    coeffs = [1.0]  # Leading Gaussian term
    
    if n_terms >= 2:
        # Skewness correction (H_3 term)
        c3 = m3 / np.sqrt(6) * np.sqrt(np.pi)
        coeffs.append(c3)
    
    if n_terms >= 3:
        # Kurtosis correction (H_4 term)
        excess_kurtosis = m4 - 3
        c4 = excess_kurtosis / 24 * np.sqrt(np.pi)
        coeffs.append(c4)
    
    return coeffs, mean, std

def edgeworth_pdf(x, coeffs, mean=0, std=1):
    """
    Evaluate Edgeworth expansion PDF.
    f(x) = φ(z) * [1 + Σ c_n He_n(z)] where z = (x-μ)/σ
    """
    from hermite import HermitePolynomial
    
    z = (x - mean) / std
    phi_z = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
    
    result = phi_z.copy()
    for n, c in enumerate(coeffs[1:], start=1):
        He_n = HermitePolynomial(n, convention="probabilist")
        result += c * He_n.evaluate(z) * phi_z
    
    return result

# Example: Approximate a non-Gaussian distribution
np.random.seed(42)
samples = np.random.gamma(shape=3, scale=1, size=10000)
coeffs, mu, sigma = edgeworth_expansion(samples, n_terms=3)
print(f"Edgeworth coefficients: {coeffs}")
```

### Example 3: Spectral Method for Differential Equations

```python
from hermite import HermiteProjection, GaussHermiteQuadrature
import numpy as np

def solve_hermite_spectral(f_rhs, n_basis=50):
    """
    Solve differential equation using Hermite spectral method.
    
    For equation: -u'' + x^2*u = f(x) (quantum harmonic oscillator)
    Uses Hermite basis which diagonalizes the operator.
    """
    # Create quadrature
    quad = GaussHermiteQuadrature(n_basis * 2, dps=50)
    roots = quad.roots
    weights = quad.weights
    
    # Project RHS onto Hermite basis
    projector = HermiteProjection(n_basis - 1)
    f_coeffs = projector.project(f_rhs)
    
    # Solve in spectral space (diagonal operator)
    u_coeffs = np.zeros(n_basis)
    for n in range(n_basis):
        # Eigenvalue of harmonic oscillator: E_n = 2*n + 1
        eigenvalue = 2 * n + 1
        if abs(eigenvalue) > 1e-10:
            u_coeffs[n] = f_coeffs[n] / eigenvalue
    
    return u_coeffs, projector

def harmonic_oscillator_rhs(x):
    """RHS function for test equation."""
    return np.exp(-x**2 / 4)

# Solve the equation
u_coeffs, projector = solve_hermite_spectral(harmonic_oscillator_rhs, n_basis=30)
x_test = np.linspace(-5, 5, 100)
u_solution = projector.reconstruct(u_coeffs, x_test)
print(f"Solution computed at {len(x_test)} points")
```

---

## Numerical Stability Notes

### Evaluation Methods

- **Clenshaw's Algorithm** is used internally for evaluating Chebyshev series — it's the most numerically stable method
- For derivatives and integrals, the library uses recurrence relations in the Hermite basis (not monomial expansion) to maintain stability

### Quadrature Accuracy

| Method | Weight Function | Best For |
|--------|-----------------|----------|
| Gauss-Hermite | e^{-x^2} | Functions decaying as Gaussian |

### Performance Considerations

- **Small n (< 64):** O(n²) direct computation is faster due to lower overhead
- **Large n (≥ 64):** FFT-based method provides O(n log n) complexity
- For very high precision needs (>100 digits), use mpmath integration

---

## API Reference Summary

```python
# Core polynomial classes
from hermite import (
    HermiteSymbolic,
    HermiteMPMath,
    HermitePolynomial
)

# Quadrature and projection
from hermite import (
    GaussHermiteQuadrature,
    HermiteProjection
)

# Utility functions
from hermite import (
    hermite_symbolic_basis,
    hermite_high_precision_basis,
    hermite_numerical_basis,
    hermite_transform,
    inverse_hermite_transform
)
```

---

## Mathematical Background

### Definition of Hermite Polynomials

**Physicist's convention (H_n):**
```
H_0(x) = 1
H_1(x) = 2x
H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
```

**Probabilist's convention (He_n):**
```
He_0(x) = 1
He_1(x) = x
He_{n+1}(x) = x·He_n(x) - n·He_{n-1}(x)
```

### Key Properties

1. **Orthogonality:** ∫_{-∞}^{+∞} H_m(x)H_n(x)e^{-x^2} dx = δ_{mn} · 2^n · n! · √π
2. **Generating function (physicist):** exp(2xt - t²) = Σ_{n=0}^{∞} H_n(x)t^n/n!
3. **Derivative:** d/dx[H_n(x)] = 2n·H_{n-1}(x)
4. **Parity:** H_n(-x) = (-1)^n · H_n(x)

---

## License and Attribution

This library is based on mathematical definitions from:
- Abramowitz & Stegun, "Handbook of Mathematical Functions"
- NIST Digital Library of Mathematical Functions (DLMF)
- Gautschi, "Orthogonal Polynomials: Computation and Approximation"

The implementation uses numerically stable algorithms adapted from established numerical libraries.
