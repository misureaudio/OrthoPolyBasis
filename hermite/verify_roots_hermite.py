# verify_roots_hermite.py
import numpy as np
from scipy.special import hermite as scipy_hermite

_cache = {}


def _get_hermite(n, use_mpmath = False):
    key = (n, use_mpmath)
    if key not in _cache:
        if use_mpmath:
            import mpmath
            _cache[key] = mpmath.hermite(n)
        else:
            _cache[key] = scipy_hermite(n)
    return _cache[key]


def verify_root(n, root, use_mpmath = False):
    if n == 0:
        return {"degree": 0, "error": "H_0(x) = 2 has no roots", "passed": False}

    H_nm1 = _get_hermite(n - 1, use_mpmath)
    H_np1 = _get_hermite(n + 1, use_mpmath)
    val_nm1 = H_nm1(root)
    val_np1 = H_np1(root)
    ratio = val_np1 / val_nm1
    expected = -2 * n
    rel_error = abs(ratio - expected) / abs(expected)

    return {
        "degree": n, "root": root,
        "expected_ratio": float(expected),
        "computed_ratio": float(ratio),
        "relative_error": float(rel_error),
        "passed": rel_error < 1e-10
    }


def verify_roots(n, roots, use_mpmath = False):
    roots = np.asarray(roots)
    results = [verify_root(n, r, use_mpmath) for r in roots]
    errors = [r["relative_error"] for r in results if "relative_error" in r]
    passed = sum(1 for r in results if r.get("passed", False))

    return {
        "degree": n, "num_roots": len(roots),
        "passed": passed, "failed": len(roots) - passed,
        "expected_ratio": float(-2 * n),
        "max_relative_error": float(max(errors)) if errors else None,
        "mean_relative_error": float(np.mean(errors)) if errors else None,
        "per_root": results
    }


if __name__ == "__main__":
    from numpy.polynomial.hermite import hermgauss
    n = 51
    roots, _ = hermgauss(n)
    print(f"Verifying {n} Hermite roots...")
    result = verify_roots(n, roots)
    print(f"Expected ratio: {result['expected_ratio']}")
    print(f"Passed: {result['passed']}/{result['num_roots']}")
    print(f"Max rel error: {result['max_relative_error']:.2e}")
