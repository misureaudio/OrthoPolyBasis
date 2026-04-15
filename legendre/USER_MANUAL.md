# LEGENDRE POLYNOMIAL LIBRARY
# ===========================
# User Manual v1.0
#
# A comprehensive Python library for Legendre polynomials with support for:
# - Coefficient generation (monomial basis)
# - Numerical evaluation
# - Symbolic computation
# - High-precision arithmetic
# - Gauss-Legendre quadrature
# - Stable operations in Legendre basis

## TABLE OF CONTENTS
# =================
# 1. Quick Start
# 2. Module Overview and Relationships
# 3. Choosing the Right Module
# 4. Detailed Module Documentation
#    4.1 core.py - Core Functionality
#    4.2 numpy_integration.py - NumPy Integration
#    4.3 sympy_integration.py - Symbolic Computation
#    4.4 mpmath_integration.py - High Precision
#    4.5 quadrature.py - Gauss-Legendre Quadrature
#    4.6 stable_operations.py - Legendre Basis Operations
#    4.7 clenshaw_evaluation.py - Series Evaluation
# 5. Common Workflows
# 6. Numerical Stability Guide
# 7. API Reference

## 1. QUICK START
# ===============

from legendre import (
    legendre_polynomial,           # Evaluate P_n(x)
    legendre_coefficients_descending,  # Get coefficients
    gauss_legendre,                # Quadrature nodes/weights
)

# Evaluate P_5(0.5)
value = legendre_polynomial(5, 0.5)

# Get coefficients of P_3(x) = x³ - (3/5)x
coeffs = legendre_coefficients_descending(3)  # [1.0, 0.0, -0.6, 0.0]

# Gauss-Legendre quadrature with 10 points
nodes, weights = gauss_legendre(10)

## 2. MODULE OVERVIEW AND RELATIONSHIPS
# =====================================

### Module Dependency Graph
#
#                    ┌─────────────┐
#                    │   core.py   │  (foundation - no dependencies)
#                    └──────┬──────┘
#                           │
#        ┌──────────────────┼──────────────────┐
#        ▼                  ▼                  ▼
#  ┌───────────┐     ┌───────────┐      ┌──────────┐
#  │numpy_int. │     │sympy_int. │      │mpmath_int│
#  └──────┬────┘     └──────┬────┘      └────┬─────┘
#         │                 │                 │
#         └─────────────────┼─────────────────┘
#                           ▼
#                    ┌──────────────┐
#                    │quadrature.py │  (uses core.py)
#                    └───────┬──────┘
#                            │
#         ┌──────────────────┼──────────────────┐
#         ▼                  ▼                  ▼
#  ┌────────────┐   ┌──────────────┐   ┌──────────────────┐
#  │stable_ops. │   │clenshaw_eval.│   │quad_high_prec.py │
#  └────────────┘   └──────────────┘   └──────────────────┘
#

### Module Categories
#
# | Category           | Modules                                    |
# |--------------------|--------------------------------------------|
# | Core               | core.py                                    |
# | Backend Integrations| numpy_integration.py, sympy_integration.py,│
# |                    | mpmath_integration.py                      |
# | Quadrature         | quadrature.py, quadrature_high_precision.py│
# | Advanced Operations| stable_operations.py, clenshaw_evaluation.py|

### Key Relationships
#
# 1. COEFFICIENT ORDER CONVENTION (IMPORTANT!)
#    All modules use DESCENDING order by default:
#        P_n(x) = c_0*x^n + c_1*x^(n-1) + ... + c_n
#    Use *_ascending() variants for: c_0 + c_1*x + ... + c_n*x^n
#
# 2. MONOMIAL vs LEGENDRE BASIS
#    - core.py, numpy_integration.py: Return monomial coefficients
#    - stable_operations.py, clenshaw_evaluation.py: Work in Legendre basis
#    - sympy_integration.py: Symbolic expressions (basis-agnostic)
#
# 3. PRECISION LEVELS
#    - core.py: Standard float64 (fastest, ~15 digits)
#    - numpy_integration.py: NumPy floats (vectorized)
#    - mpmath_integration.py: Arbitrary precision (slowest, unlimited digits)
#    - sympy_integration.py: Exact rational arithmetic

## 3. CHOOSING THE RIGHT MODULE
# =============================

### Decision Tree for Coefficient Generation
#
# What do you need?
# ├─ Symbolic/exact results? → sympy_integration.py
# ├─ High precision (>15 digits)? → mpmath_integration.py
# ├─ Vectorized/arrays? → numpy_integration.py
# └─ Simple/fast? → core.py

### Decision Tree for Evaluation
#
# How will you evaluate?
# ├─ Single P_n(x) at point(s)?
# │  └─ Use core.LegendreGenerator.evaluate() or legendre_polynomial()
# ├─ Linear combination Σ a_k * P_k(x)?
# │  └─ Use clenshaw_evaluation.evaluate_legendre_series()
# ├─ Need derivative/integral?
# │  ├─ Symbolic? → sympy_integration.py
# │  └─ Numerical? → stable_operations.py (for high n)
# └─ Arbitrary precision needed?
#    └─ mpmath_integration.py

### Decision Tree for Quadrature
#
# What are you integrating?
# ├─ Standard double precision, n < 50?
# │  └─ quadrature.gauss_legendre(n) [Golub-Welsch]
# ├─ High n (>50) or need stability?
# │  └─ quadrature.GaussLegendreQuadrature.golub_welsch(n)
# ├─ Need >15 digit precision?
# │  └─ quadrature_high_precision.gauss_legendre_high_precision(n, dps=50)
# └─ Just need roots of P_n?
#    └─ Any quadrature method; take the nodes

## 4. DETAILED MODULE DOCUMENTATION
# ================================

### 4.1 core.py - Core Functionality
#
# Purpose: Fast, simple Legendre polynomial operations in standard float64.
#
# Best for: Quick computations, n < 60, when dependencies should be minimal.
#
# Key Classes/Functions:
#   LegendreGenerator()
#     .get_coefficients_descending(n)  → [c_0, c_1, ..., c_n]
#     .get_coefficients_ascending(n)   → [c_n, c_{n-1}, ..., c_0]
#     .evaluate(x, n)                  → P_n(x)
#     .derivative_coefficients(n)      → coefficients of P_n'
#     .integral_coefficients(n)        → coefficients of ∫P_n
#
# Convenience functions:
#   legendre_polynomial(n, x)          → P_n(x)
#   legendre_coefficients_descending(n)
#   legendre_coefficients_ascending(n)
#
# Example:
#   from legendre import legendre_polynomial, legendre_coefficients_descending
#   
#   value = legendre_polynomial(5, 0.5)        # P_5(0.5)
#   coeffs = legendre_coefficients_descending(3)  # [1.0, 0.0, -0.6, 0.0]
#
# Stability Notes:
#   ✓ evaluate() uses stable three-term recurrence (safe for any n)
#   ✗ Coefficients become unstable for n > 60
#   ✗ derivative_coefficients()/integral_coefficients() unstable for n > 60

### 4.2 numpy_integration.py - NumPy Integration
#
# Purpose: Vectorized operations using NumPy arrays.
#
# Best for: Batch processing, array inputs, integration with NumPy ecosystem.
#
# Key Class:
#   NumpyLegendreGenerator()
#     .get_coefficients_descending(n)  → np.ndarray (descending)
#     .get_coefficients_ascending(n)   → np.ndarray (ascending)
#     .evaluate(x, n)                  → P_n(x), x can be array!
#     .evaluate_batch(xs, n)           → Evaluate at multiple points
#
# Convenience functions:
#   generate_numpy_legendre_descending(n)
#   generate_numpy_legendre_ascending(n)
#   evaluate_numpy_legendre(x, n)
#
# Example:
#   from legendre import generate_numpy_legendre_descending, evaluate_numpy_legendre
#   import numpy as np
#   
#   coeffs = generate_numpy_legendre_descending(5)  # numpy array
#   xs = np.linspace(-1, 1, 100)
#   values = evaluate_numpy_legendre(xs, 5)         # vectorized!
#
# Relationship to core.py:
#   - Same algorithms, returns numpy arrays instead of lists
#   - Uses numpy.polynomial.legendre.legval for stable evaluation
#   - Coefficient order convention matches core.py

### 4.3 sympy_integration.py - Symbolic Computation
#
# Purpose: Exact symbolic manipulation using SymPy.
#
# Best for: Derivations, exact arithmetic, symbolic differentiation/integration.
#
# Key Class:
#   SympyLegendreGenerator()
#     .get_polynomial(n)           → SymPy expression for P_n(x)
#     .get_derivative(n)           → Symbolic derivative
#     .get_integral(n)             → Symbolic integral
#     .evaluate(x_val, n)          → Numeric evaluation of symbolic form
#
# Convenience functions:
#   generate_sympy_legendre(n)
#   get_sympy_legendre_basis(max_n)
#
# Example:
#   from legendre import generate_sympy_legendre
#   from sympy import diff, integrate
#   
#   P3 = generate_sympy_legendre(3)        # Symbolic: x*(3*x**2 - 1)/2
#   dP3 = diff(P3, x)                      # Symbolic derivative
#   int_P3 = integrate(P3, x)              # Symbolic integral
#   value = P3.subs(x, 0.5)                # Evaluate at x=0.5
#
# Relationship to other modules:
#   - Only module providing exact rational arithmetic
#   - Slowest but most precise
#   - Can convert to numeric via .evalf() or .subs()

### 4.4 mpmath_integration.py - High Precision
#
# Purpose: Arbitrary precision arithmetic using mpmath.
#
# Best for: n > 60 with coefficient needs, high-precision evaluation.
#
# Key Class:
#   MpmathLegendreGenerator()
#     .get_coefficients_descending(n)  → List[mp.mpf]
#     .get_coefficients_ascending(n)   → List[mp.mpf]
#     .evaluate(x, n, dps=None)        → mp.mpf with specified precision
#     .get_polynomial(n)               → Callable using mpmath.legendre
#
# Convenience functions:
#   get_mpmath_legendre_coefficients_descending(n)
#   get_mpmath_legendre_coefficients_ascending(n)
#   evaluate_mpmath_legendre(x, n, dps=50)
#
# Example:
#   from legendre import get_mpmath_legendre_coefficients_descending
#   
#   # 50 decimal places of precision!
#   coeffs = get_mpmath_legendre_coefficients_descending(100)
#   print(coeffs[0])  # mpf('2.679...e+29') with full precision
#
# Relationship to core.py:
#   - Same algorithms but with mp.mpf instead of float
#   - Respects global mp.dps setting
#   - Much slower but arbitrary precision

### 4.5 quadrature.py - Gauss-Legendre Quadrature
#
# Purpose: Compute nodes and weights for numerical integration.
#
# Best for: Numerical integration, finding roots of P_n(x).
#
# Key Class:
#   GaussLegendreQuadrature
#     .newton_raphson(n)              → (nodes, weights), good for n < 50
#     .golub_welsch(n)                → (nodes, weights), stable for any n
#     .integrate(f, nodes, weights)   → ∫_{-1}^{1} f(x) dx
#     .integrate_transformed(f, a, b, nodes, weights)  → ∫_a^b f(x) dx
#
# Convenience functions:
#   gauss_legendre(n, method="golub_welsch")
#   gauss_legendre_newton(n)
#   gauss_legendre_golub_welsch(n)
#
# Example:
#   from legendre import gauss_legendre, GaussLegendreQuadrature
#   import numpy as np
#   
#   # Get 20-point quadrature rule
#   nodes, weights = gauss_legendre(20)
#   
#   # Integrate exp(x) from -1 to 1
#   result = GaussLegendreQuadrature.integrate(np.exp, nodes, weights)
#   
#   # Integrate sin(x) from 0 to π
#   result = GaussLegendreQuadrature.integrate_transformed(
#       np.sin, 0, np.pi, nodes, weights
#   )
#
# Relationship to core.py:
#   - Uses core._evaluate_legendre() internally for Newton-Raphson
#   - Golub-Welsch is eigenvalue-based (no root-finding)
#
# Important Properties:
#   - n-point rule exactly integrates polynomials of degree 2n-1
#   - Sum of weights always equals 2.0 (interval length)
#   - Nodes are roots of P_n(x)

### 4.6 stable_operations.py - Legendre Basis Operations
#
# Purpose: Operations that stay in Legendre basis (avoid monomial instability).
#
# Best for: High-degree polynomial manipulation, derivatives/integrals.
#
# Key Class:
#   LegendreBasisOperations
#     .derivative_legendre_basis(coeffs)    → P' in Legendre basis
#     .integral_legendre_basis(coeffs, C)   → ∫P in Legendre basis  
#     .multiply_by_x_legendre_basis(coeffs) → x·P in Legendre basis
#     .convert_from_monomial(mono_coeffs)   → monomial → Legendre
#     .convert_to_monomial(leg_coeffs)      → Legendre → monomial
#
# Convenience functions:
#   derivative_legendre(coeffs)
#   integral_legendre(coeffs, C=0.0)
#
# Example:
#   from legendre.stable_operations import LegendreBasisOperations
#   import numpy as np
#   
#   # Polynomial: 2*P_0 + 3*P_1 - P_2 (in Legendre basis!)
#   coeffs = np.array([2.0, 3.0, -1.0])
#   
#   # Derivative stays in Legendre basis
#   deriv_coeffs = LegendreBasisOperations.derivative_legendre_basis(coeffs)
#   
#   # Integral with constant C=1
#   int_coeffs = LegendreBasisOperations.integral_legendre_basis(coeffs, C=1.0)
#
# Relationship to other modules:
#   - Does NOT return monomial coefficients like core.py
#   - Works with Legendre-basis coefficients: p(x) = Σ c_k * P_k(x)
#   - Much more stable than core.derivative_coefficients() for high n
#   - Use clenshaw_evaluation.evaluate_legendre_series() to evaluate results

### 4.7 clenshaw_evaluation.py - Series Evaluation
#
# Purpose: Efficiently evaluate Σ a_k * P_k(x) using Clenshaw algorithm.
#
# Best for: Evaluating linear combinations of Legendre polynomials.
#
# Key Function:
#   evaluate_legendre_series(coeffs, x)
#
# Example:
#   from legendre.clenshaw_evaluation import evaluate_legendre_series
#   import numpy as np
#   
#   # Evaluate: 1*P_0(x) + 2*P_1(x) + 3*P_2(x) at x = 0.5
#   coeffs = np.array([1.0, 2.0, 3.0])  # Legendre basis coefficients!
#   result = evaluate_legendre_series(coeffs, 0.5)
#
# Relationship to other modules:
#   - Complements stable_operations.py (evaluate what it computes)
#   - Coefficients are in LEGENDRE basis, not monomial basis
#   - O(n) algorithm, numerically stable

## 5. COMMON WORKFLOWS
# ====================

### Workflow 1: Basic Evaluation
#
from legendre import legendre_polynomial
value = legendre_polynomial(10, 0.5)  # P_10(0.5)

### Workflow 2: Get Coefficients and Evaluate Polynomial
#
# WARNING: Don't use coefficients for evaluation at high n!
from legendre import legendre_coefficients_descending
import numpy as np

coeffs = legendre_coefficients_descending(5)  # [1, 0, -9/4, 0, 5/8, 0]
x = 0.5
value = np.polyval(coeffs, x)  # OK for n < 60

### Workflow 3: High-Degree Evaluation (RECOMMENDED)
#
from legendre import legendre_polynomial
value = legendre_polynomial(100, 0.5)  # Uses stable recurrence!

### Workflow 4: Symbolic Manipulation
#
from legendre import generate_sympy_legendre
from sympy import symbols, diff, integrate, simplify

x = symbols("x")
P5 = generate_sympy_legendre(5)
dP5 = diff(P5, x)           # Symbolic derivative
int_P5 = integrate(P5, x)   # Symbolic integral
simplified = simplify(dP5)  # Simplify result

### Workflow 5: Numerical Integration via Quadrature
#
from legendre import gauss_legendre, GaussLegendreQuadrature
import numpy as np

# Integrate exp(-x²) from -∞ to +∞ (approximated as [-10, 10])
nodes, weights = gauss_legendre(50)
f = lambda x: np.exp(-x*x)
result = GaussLegendreQuadrature.integrate_transformed(f, -10, 10, nodes, weights)
# Result ≈ √π = 1.77245...

### Workflow 6: High-Degree Polynomial Operations
#
from legendre.stable_operations import LegendreBasisOperations
from legendre.clenshaw_evaluation import evaluate_legendre_series
import numpy as np

# Define p(x) = Σ_{k=0}^{100} (1/k!) * P_k(x) in Legendre basis
coeffs = np.array([1.0/np.math.factorial(k) for k in range(101)])

# Take derivative (stays in Legendre basis, stable!)
deriv_coeffs = LegendreBasisOperations.derivative_legendre_basis(coeffs)

# Evaluate at x = 0.5
value = evaluate_legendre_series(deriv_coeffs, 0.5)

### Workflow 7: High-Precision Coefficients
#
from legendre import get_mpmath_legendre_coefficients_descending

# Get P_100 coefficients with 50 decimal places
coeffs = get_mpmath_legendre_coefficients_descending(100)
print(coeffs[0])  # Full precision!

## 6. NUMERICAL STABILITY GUIDE
# ==============================

### When n < 30:
#   All methods work well. Use core.py for simplicity.

### When 30 ≤ n < 60:
#   ✓ Evaluation: Any method (all use stable recurrence)
#   ⚠ Coefficients: Precision loss begins, but often acceptable
#   ⚗ Derivatives/Integrals: Use stable_operations.py

### When n ≥ 60:
#   ✓ Evaluation: core.evaluate() or legendre_polynomial()
#   ✗ Coefficients (float): Avoid! Use mpmath or sympy
#   ✗ Monomial derivatives/integrals: Use stable_operations.py
#   ✓ Quadrature: Golub-Welsch method

### Precision Comparison:
#
# | Method              | Precision     | Speed    | Max Practical n |
# |---------------------|---------------|----------|------------------|
# | core.py (eval)      | ~15 digits    | Fastest  | ∞                |
# | core.py (coeffs)    | ~15 digits    | Fastest  | ~60              |
# | numpy_integration   | ~15 digits    | Fast     | ∞ (eval), ~60    |
# | sympy_integration   | Exact         | Slowest  | ~200             |
# | mpmath_integration  | User-defined  | Slow     | ∞                |
# | stable_operations   | ~15 digits    | Fast     | ∞                |

## 7. API REFERENCE
# =================

### Module: legendre.core
#
# Class: LegendreGenerator
#   Methods:
#     get_coefficients_descending(n: int) -> List[float]
#     get_coefficients_ascending(n: int) -> List[float] 
#     evaluate(x: float, n: int) -> float
#     evaluate_batch(xs: List[float], n: int) -> List[float]
#     derivative_coefficients(n: int) -> List[float]
#     integral_coefficients(n: int) -> List[float]
#     generate_basis_descending(max_n: int) -> List[List[float]]
#     generate_basis_ascending(max_n: int) -> List[List[float]]
#
# Functions:
#   legendre_polynomial(n: int, x: float) -> float
#   legendre_coefficients_descending(n: int) -> List[float]
#   legendre_coefficients_ascending(n: int) -> List[float]
#   legendre_derivative(n: int) -> List[float]
#   legendre_integral(n: int) -> List[float]

### Module: legendre.numpy_integration
#
# Class: NumpyLegendreGenerator
#   Methods:
#     get_coefficients_descending(n: int) -> np.ndarray
#     get_coefficients_ascending(n: int) -> np.ndarray
#     evaluate(x: Union[float, np.ndarray], n: int)
#     evaluate_batch(xs: np.ndarray, n: int) -> np.ndarray
#
# Functions:
#   generate_numpy_legendre_descending(n: int) -> np.ndarray
#   generate_numpy_legendre_ascending(n: int) -> np.ndarray
#   evaluate_numpy_legendre(x, n: int)

### Module: legendre.sympy_integration
#
# Class: SympyLegendreGenerator  
#   Methods:
#     get_polynomial(n: int) -> sympy.Expr
#     get_derivative(n: int) -> sympy.Expr
#     get_integral(n: int) -> sympy.Expr
#     evaluate(x_val, n: int) -> float
#
# Functions:
#   generate_sympy_legendre(n: int) -> sympy.Expr
#   get_sympy_legendre_basis(max_n: int) -> List[sympy.Expr]

### Module: legendre.mpmath_integration
#
# Class: MpmathLegendreGenerator
#   Methods:
#     get_coefficients_descending(n: int) -> List[mp.mpf]
#     get_coefficients_ascending(n: int) -> List[mp.mpf]
#     evaluate(x: str, n: int, dps: Optional[int]) -> mp.mpf
#     get_polynomial(n: int) -> Callable
#
# Functions:
#   get_mpmath_legendre_coefficients_descending(n: int) -> List[mp.mpf]
#   get_mpmath_legendre_coefficients_ascending(n: int) -> List[mp.mpf]
#   evaluate_mpmath_legendre(x: str, n: int, dps: int)

### Module: legendre.quadrature
#
# Class: GaussLegendreQuadrature
#   Static Methods:
#     newton_raphson(n: int) -> Tuple[np.ndarray, np.ndarray]
#     golub_welsch(n: int) -> Tuple[np.ndarray, np.ndarray]
#     integrate(f, nodes, weights) -> float
#     integrate_transformed(f, a, b, nodes, weights) -> float
#
# Functions:
#   gauss_legendre(n: int, method="golub_welsch")
#   gauss_legendre_newton(n: int)
#   gauss_legendre_golub_welsch(n: int)

### Module: legendre.quadrature_high_precision
#
# Class: HighPrecisionGaussLegendre(dps: int = 50)
#   Methods:
#     compute(n: int) -> Tuple[List[mp.mpf], List[mp.mpf]]
#     integrate(f, n: int) -> mp.mpf
#
# Functions:
#   gauss_legendre_high_precision(n: int, dps: int = 50)

### Module: legendre.stable_operations
#
# Class: LegendreBasisOperations
#   Static Methods:
#     derivative_legendre_basis(coeffs: np.ndarray) -> np.ndarray
#     integral_legendre_basis(coeffs: np.ndarray, C: float) -> np.ndarray
#     multiply_by_x_legendre_basis(coeffs: np.ndarray) -> np.ndarray
#     convert_from_monomial(mono_coeffs: np.ndarray) -> np.ndarray
#     convert_to_monomial(leg_coeffs: np.ndarray) -> np.ndarray
#
# Functions:
#   derivative_legendre(coeffs: np.ndarray) -> np.ndarray
#   integral_legendre(coeffs: np.ndarray, C: float = 0.0) -> np.ndarray

### Module: legendre.clenshaw_evaluation
#
# Functions:
#   evaluate_legendre_series(coeffs: np.ndarray, x) -> Union[float, np.ndarray]

## APPENDIX A: COEFFICIENT ORDER EXAMPLES
# ======================================

P_2(x) = (3x² - 1)/2 = 1.5*x² + 0*x - 0.5

Descending order [1.5, 0.0, -0.5]:
  Index:     0      1      2
  Power:     2      1      0
  Meaning:   1.5*x² + 0*x¹ + (-0.5)*x⁰

Ascending order [-0.5, 0.0, 1.5]:
  Index:     0      1      2  
  Power:     0      1      2
  Meaning:   -0.5*x⁰ + 0*x¹ + 1.5*x²

## APPENDIX B: LEGENDRE BASIS vs MONOMIAL BASIS
# ==============================================

Same polynomial, two representations:

Monomial basis (from core.py):
  p(x) = 1.5*x² + 2.0*x - 0.5
  coeffs = [1.5, 2.0, -0.5]  # coefficients of x², x¹, x⁰

Legendre basis (for stable_operations.py):
  p(x) = a₀*P₀(x) + a₁*P₁(x) + a₂*P₂(x)
       = a₀*1 + a₁*x + a₂*(3x²-1)/2
  coeffs = [a₀, a₁, a₂]  # coefficients of P₀, P₁, P₂

To convert between them:
  from legendre.stable_operations import LegendreBasisOperations
  leg_coeffs = LegendreBasisOperations.convert_from_monomial(mono_coeffs)
  mono_coeffs = LegendreBasisOperations.convert_to_monomial(leg_coeffs)
