# Chebyshev Polynomial Generator

A Python module for generating Chebyshev polynomials, designed to work seamlessly with:

- **SymPy** - Symbolic mathematics
- **NumPy** - Numerical arrays and computation  
- **mpmath** - Arbitrary precision arithmetic

## Installation

```bash
pip install numpy sympy mpmath scipy
```

## Usage

### Core Module (Pure Python)

```python
from chebyshev.core import ChebyshevGenerator

gen = ChebyshevGenerator()
coeffs = gen.get_coefficients(3)  # [4.0, 0.0, -3.0, 0.0]
value = gen.evaluate(0.5, 3)      # T_3(0.5) = -0.5
basis = gen.generate_basis(5)     # Generate T_0 through T_5
```

### SymPy Integration (Symbolic)

```python
from chebyshev.sympy_integration import generate_sympy_chebyshev
poly = generate_sympy_chebyshev(3)  # 8*x**3 - 6*x
deriv = poly.diff('x')              # Derivative
integ = poly.integrate('x')         # Integral
```

### NumPy Integration (Numerical)

```python
from chebyshev.numpy_integration import generate_numpy_chebyshev, evaluate_numpy_chebyshev
import numpy as np
coeffs = generate_numpy_chebyshev(3)
xs = np.array([-1, -0.5, 0, 0.5, 1])
values = evaluate_numpy_chebyshev(xs, 3)
```

### mpmath Integration (Arbitrary Precision)

```python
from chebyshev.mpmath_integration import evaluate_mpmath_chebyshev
result = evaluate_mpmath_chebyshev('0.123456789', 5, dps=50)
```

## Chebyshev Polynomials Reference

| n | T_n(x) |
|---|--------|
| 0 | 1 |
| 1 | x |
| 2 | 2x^2 - 1 |
| 3 | 4x^3 - 3x |
| 4 | 8x^4 - 8x^2 + 1 |

## Properties

- **Definition**: T_n(x) = cos(n * arccos(x)) for x ∈ [-1, 1]
- **Recurrence**: T_0(x) = 1, T_1(x) = x, T_n(x) = 2x·T_{n-1}(x) - T_{n-2}(x)
- **Orthogonality**: Weighted orthogonality with weight w(x) = 1/√(1-x²) over [-1,1]
- **Boundary values**: T_n(1) = 1, T_n(-1) = (-1)^n
- **Extrema**: T_n(x) oscillates between -1 and 1 on [-1, 1]

## Running Tests

```bash
cd chebyshev
python tests.py