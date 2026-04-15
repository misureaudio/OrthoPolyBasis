# verify_roots_laguerre.py

import numpy as np
from scipy.special import laguerre as scipy_laguerre

_cache = {}


def _get_laguerre(n, alpha, use_mpmath):
    key = (n, alpha, use_mpmath)
    if key not in _cache:
        if use_mpmath:
            import mpmath
            _cache[key] = mpmath.laguerre(n, alpha)
        else:
            _cache[key] = scipy_laguerre(n, alpha)
    return _cache[key]


def verify_root(n, root, alpha = 0.0, use_mpmath = False):
    if n == 0:
        return {"degree": 0, "error": "L_0(x) = 1 has no roots", "passed": False}

    L_nm1 = _get_laguerre(n - 1, alpha, use_mpmath)
    L_np1 = _get_laguerre(n + 1, alpha, use_mpmath)
    val_nm1 = L_nm1(root)
    val_np1 = L_np1(root)
    ratio = val_np1 / val_nm1
    expected = -(n + alpha) / (n + 1) # Laguerre ratio property
    rel_error = abs(ratio - expected) / abs(expected)

    return {
        "degree": n, "root": root, "alpha": alpha,
        "expected_ratio": float(expected),
        "computed_ratio": float(ratio),
        "relative_error": float(rel_error),
        "passed": rel_error < 1e-10
    }


def verify_roots(n, roots, alpha = 0.0, use_mpmath = False):
    roots = np.asarray(roots)
    results = [verify_root(n, r, alpha, use_mpmath) for r in roots]
    errors = [r["relative_error"] for r in results if "relative_error" in r]
    passed = sum(1 for r in results if r.get("passed", False))

    return {
        "degree": n, "num_roots": len(roots), "alpha": alpha,
        "passed": passed, "failed": len(roots) - passed,
        "expected_ratio": float((-n - alpha) / (n + 1)),
        "max_relative_error": float(max(errors)) if errors else None,
        "mean_relative_error": float(np.mean(errors)) if errors else None,
        "per_root": results
    }


if __name__ == "__main__":
    from numpy.polynomial.laguerre import laggauss
    n, alpha = 51, 0.0
    roots, _ = laggauss(n)
    print(f"Verifying {n} Laguerre roots alpha={alpha}...")
    result = verify_roots(n, roots, alpha)
    print(f"Expected ratio: {result['expected_ratio']}")
    print(f"Passed: {result['passed']}/{result['num_roots']}")
    print(f"Max rel error: {result['max_relative_error']:.2e}")
