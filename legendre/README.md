# Legendre Polynomial Generator

A Python module for generating Legendre polynomials, designed to work seamlessly with:

- **SymPy** - Symbolic mathematics
- **NumPy** - Numerical arrays and computation  
- **mpmath** - Arbitrary precision arithmetic

## Installation

```bash
pip install numpy sympy mpmath scipy
```

## Usage

### Core Module (Pure Python)

```Python
from legendre.core import LegendreGenerator

gen = LegendreGenerator()
coeffs = gen.get_coefficients(3)  # [2.5, 0.0, -1.5, 0.0]
value = gen.evaluate(0.5, 3)
basis = gen.generate_basis(5)
```

### SymPy Integration (Symbolic)

```Python
from legendre.sympy_integration import generate_sympy_legendre
poly = generate_sympy_legendre(3)  # (5*x**3 - 3*x)/2
deriv = poly.diff('x')
integ = poly.integrate('x')
```

### NumPy Integration (Numerical)

```Python
from legendre.numpy_integration import generate_numpy_legendre, evaluate_numpy_legendre
import numpy as np
coeffs = generate_numpy_legendre(3)
xs = np.array([-1, -0.5, 0, 0.5, 1])
values = evaluate_numpy_legendre(xs, 3)
```

### mpmath Integration (Arbitrary Precision)

```Python
from legendre.mpmath_integration import evaluate_mpmath_legendre
result = evaluate_mpmath_legendre('0.123456789', 5, dps=50)
```

## Legendre Polynomials Reference

| n | P_n(x) |
|---|--------|
| 0 | 1 |
| 1 | x |
| 2 | (3x^2 - 1)/2 |
| 3 | (5x^3 - 3x)/2 |
| 4 | (35x^4 - 30x^2 + 3)/8 |

## Properties

- **Orthogonality**: integral of P_n*P_m over [-1,1] = 0 for n != m
- **Normalization**: integral of P_n^2 over [-1,1] = 2/(2n+1)
- **Boundary values**: P_n(1) = 1, P_n(-1) = (-1)^n

## Running Tests

```bash
cd legendre
python tests.py
```
