# Detailed Implementation Plan: Root Verification Modules

## Overview

  Create four verification modules, one for each polynomial family, implementing the numerically stable ratio property
  test documented in OrthoPoly_HDEG_CALC.md.

  ———

## Phase 1: Design Decisions & Architecture

### 1.1 Module Structure (per family)

  verify_roots_<family>.py
  ├── verify_root(n, root, use_mpmath=False) → dict
  │   └── Verify single root, return {ratio, expected, rel_error}
  ├── verify_roots(n, roots, use_mpmath=False) → dict
  │   └── Batch verification, return stats + per-root details
  └── CLI interface (if __name__ == "__main__")

### 1.2 Ratio Properties to Implement

| Family | Function(s) | Expected Ratio at Root |
|--------|-------------|----------------------|
| hermite | scipy.special.hermite(n) | $H_{n+1}/H_{n-1} = -2n$ |
| chebyshev | scipy.special.chebyt(n) | $T_{n+1}/T_{n-1} = -1$ |
| legendre | scipy.special.legendre(n) | $P_{n+1}/P_{n-1} = -n/(n+1)$ |
| laguerre | scipy.special.laguerre(n, alpha) | $L_{n+1}^{(\alpha)}/L_{n-1}^{(\alpha)} = -(n+\alpha)/(n+1)$ |

### 1.3 Return Format (consistent across all modules)

  {
      "degree": n,
      "expected_ratio": float,
      "computed_ratio": float,
      "relative_error": float,
      "passed": bool# rel_error < 1e-10
  }

  ———

## Phase 2: Implementation Steps

### Step 1: Create verify_roots_hermite.py

- Import scipy.special.hermite (NumPy path) and mpmath.hermite (high-precision path)
- Implement verify_root(n, root, use_mpmath=False)
- Implement verify_roots(n, roots, use_mpmath=False) with statistics
- Add CLI: accept degree + root(s) from args or file

### Step 2: Create verify_roots_chebyshev.py

- Import scipy.special.chebyt (NumPy path) and mpmath.chebyt (high-precision path)
- Same interface as hermite module
- Expected ratio is constant -1 for all degrees

### Step 3: Create verify_roots_legendre.py

- Import scipy.special.legendre (NumPy path) and mpmath.legendre (high-precision path)
- Same interface, expected ratio depends on degree: $-n/(n+1)$

### Step 4: Create verify_roots_laguerre.py

- Import scipy.special.laguerre(n, alpha) with optional alpha parameter
- Expected ratio: $-(n+\alpha)/(n+1)$
- Default alpha=0 for standard Laguerre polynomials

### Step 5: Add Common Features to All Modules

- Caching: Module-level cache for polynomial objects (like existing codebase)
- Warnings: Alert if use_mpmath=False at high degrees (per family thresholds)
- Docstrings: Full documentation with mathematical formulas
- Type hints: Python type annotations

  ———

## Phase 3: Testing Strategy

### For each module, test:

1. Single root verification against known roots from numpy.polynomial.<family>.<family>gauss(n)
2. Batch verification of all roots for degrees n=10, 50, 100
3. Compare NumPy vs mpmath paths for consistency
4. Verify relative errors are within expected bounds (~1e-13 to 1e-15)

  ———

## Phase 4: Documentation

  Add a README section or docstring explaining:

- Mathematical basis (ratio property derivation)
- When to use use_mpmath=True vs False
- Expected accuracy per family
- Example usage

  ———

## Files to Create

| File | Location | Lines (est.) |
|------|----------|--------------|
| verify_roots_hermite.py | ./OrthoPolyB_np_mp/ hermite/ | ~80 |
| verify_roots_chebyshev.py | ./OrthoPolyB_np_mp/ chebyshev/ | ~75 |
| verify_roots_legendre.py | ./OrthoPolyB_np_mp/ legendre/ | ~75 |
| verify_roots_laguerre.py | ./OrthoPolyB_np_mp/ laguerre/ | ~80 |

  Total: ~310 lines across 4 modules

  ———

## Waiting for Approval

  Please confirm:

1. Module naming convention: verify_roots_<family>.py
2. Function signatures as described above
3. Return format (dictionary with ratio, expected, rel_error)
4. Include both NumPy and mpmath paths
5. Add CLI interface for each module
