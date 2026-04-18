# quadrature_analyzer.py
from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
from sympy import (
    Symbol, diff, N, S, pi, log, sin, cos, tan, exp, sqrt,
    Abs as SymAbs, limit, solve, Derivative,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

_PARSER_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

class PolynomialFamily(str, Enum):
    LEGENDRE = "Legendre"
    CHEBYSHEV = "Chebyshev"
    HERMITE = "Hermite"
    LAGUERRE = "Laguerre"

@dataclass
class FunctionAnalysis:
    original_expr: str
    sympy_expr: object
    variable: str = "x"
    interval_type: str = "finite"
    singularities: list = field(default_factory=list)
    has_endpoint_singularity: bool = False
    has_interior_singularity: bool = False
    max_deriv_order_probed: int = 6
    derivative_growth_rate: str = "bounded"
    decay_type: str = "none"
    decay_rate: float = 0.0
    is_periodic_on_interval: bool = False
    approximate_period: Optional[float] = None
    recommended_family: PolynomialFamily = PolynomialFamily.LEGENDRE
    confidence: str = "high"
    recommendation_reason: str = ""
    suggested_min_n: int = 8
    suggested_max_n: int = 64
    degree_criteria: list[str] = field(default_factory=list)


def _safe_eval_sympy(expr, x_val):
    try:
        return float(N(expr.subs("x", x_val), 15))
    except Exception:
        return float("nan")


class QuadratureAnalyzer:
    """
    Reads a mathematical expression (string or SymPy object) that defines a real
    function of one variable, analyses its properties, and recommends the optimal
    orthogonal-polynomial family together with quadrature-degree criteria.

    The recommended polynomial family maps directly to the corresponding module in
    OrthoPolyB_np_mp:
      - Legendre  -> `from legendre import LegendreQuadrature`
      - Chebyshev -> `from chebyshev import ChebyshevQuadrature`
      - Hermite   -> `from hermite import GaussHermiteQuadrature`
      - Laguerre  -> `from laguerre import LaguerreQuadrature`

    Usage:
        analyzer = QuadratureAnalyzer()
        analysis = analyzer.analyze("exp(-x**2) * sin(x)", interval=(-1, 1))
        print(analysis.recommended_family)
        print(analysis.degree_criteria)
    """

    def __init__(self, default_variable: str = "x"):
        self._var = Symbol(default_variable)

    def analyze(
        self,
        expression,
        interval=None,
        *,
        variable: str = "x",
        max_deriv_order: int = 6,
    ) -> FunctionAnalysis:
        self._var = Symbol(variable)
        expr = self._parse(expression, variable)
        if interval is None:
            interval = self._infer_interval(expr, variable)
        a, b = float(interval[0]), float(interval[1])
        analysis = FunctionAnalysis(
            original_expr=str(expression), sympy_expr=expr, variable=variable,
        )
        analysis.interval_type = self._classify_interval(a, b)
        analysis.singularities = self._find_singularities(expr, a, b)
        analysis.has_endpoint_singularity = any(
            s["location"] in (a, b) for s in analysis.singularities
        )
        analysis.has_interior_singularity = any(
            a < s["location"] < b for s in analysis.singularities
        )
        analysis.derivative_growth_rate = self._probe_derivative_growth(expr, a, b, max_deriv_order)
        if analysis.interval_type in ("semi_infinite", "infinite"):
            analysis.decay_type, analysis.decay_rate = self._probe_decay(expr, variable)
        if a > -1e9 and b < 1e9:
            analysis.is_periodic_on_interval, analysis.approximate_period = (
                self._probe_periodicity(expr, a, b)
            )
        rec = self._recommend_family(analysis, a, b)
        analysis.recommended_family = rec.family
        analysis.confidence = rec.confidence
        analysis.recommendation_reason = rec.reason
        analysis.degree_criteria = self._degree_criteria(analysis)
        lo, hi = self._suggest_degree_range(analysis)
        analysis.suggested_min_n = lo
        analysis.suggested_max_n = hi
        return analysis

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse(expression, variable: str) -> object:
        if isinstance(expression, str):
            return parse_expr(
                expression,
                local_dict={variable: Symbol(variable)},
                transformations=_PARSER_TRANSFORMS,
            )
        return expression

    @staticmethod
    def _infer_interval(expr, variable: str) -> tuple:
        v = Symbol(variable)
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if inner == -v ** 2 or inner == -(v ** 2):
                    return (-float("inf"), float("inf"))
            except Exception:
                pass
        for sub in expr.atoms(log):
            try:
                arg = sub.args[0]
                if arg == v:
                    return (0.0, float("inf"))
            except Exception:
                pass
        return (-1.0, 1.0)

    @staticmethod
    def _classify_interval(a: float, b: float) -> str:
        if a == -float("inf") and b == float("inf"):
            return "infinite"
        if a == -float("inf") or b == float("inf"):
            return "semi_infinite"
        return "finite"

    def _find_singularities(self, expr, a: float, b: float) -> list:
        v = self._var
        found: list = []
        try:
            den = expr.as_numer_denom()[1]
            if den != 1:
                roots = solve(den, v, domain=S.Complexes)
                for r in roots:
                    try:
                        rv = float(N(r, 15))
                        if math.isnan(rv):
                            continue
                        if a <= rv <= b:
                            found.append({"location": rv, "kind": "pole"})
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
        for sub in expr.atoms(log):
            try:
                arg = sub.args[0]
                zero_pts = solve(arg, v)
                for zp in zero_pts:
                    try:
                        zv = float(N(zp, 15))
                        if a <= zv <= b and not math.isnan(zv):
                            found.append({"location": zv, "kind": "log_zero"})
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        for sub in expr.atoms(sqrt):
            try:
                arg = sub.args[0]
                zero_pts = solve(arg, v)
                for zp in zero_pts:
                    try:
                        zv = float(N(zp, 15))
                        if a <= zv <= b and not math.isnan(zv):
                            found.append({"location": zv, "kind": "sqrt_zero"})
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass
        for sub in expr.atoms(tan):
            try:
                arg = sub.args[0]
                half_pi = float(pi / 2)
                for k in range(-10, 11):
                    sing_val = half_pi + k * float(pi)
                    if a <= sing_val <= b:
                        found.append({"location": sing_val, "kind": "tan_pole"})
            except Exception:
                pass
        unique: list = []
        for s in found:
            if not any(abs(s["location"] - u["location"]) < 1e-8 for u in unique):
                unique.append(s)
        return unique

    def _probe_derivative_growth(self, expr, a: float, b: float, max_order: int) -> str:
        v = self._var
        mid = (a + b) / 2.0 if not np.isinf(a) and not np.isinf(b) else 0.0
        growths: list = []
        for k in range(1, min(max_order + 1, 7)):
            dexpr = diff(expr, v, k)
            try:
                val = float(N(dexpr.subs(v, mid), 10))
                if math.isfinite(val):
                    growths.append(abs(val))
            except Exception:
                break
        if len(growths) < 2:
            return "bounded"
        ratios: list = []
        for i in range(1, len(growths)):
            if growths[i - 1] > 0 and growths[i] > 0:
                ratios.append(math.log(growths[i]) / max(math.log(growths[i - 1]), 1e-30))
        if not ratios:
            return "bounded"
        avg_ratio = np.mean(ratios)
        if avg_ratio < 2.5:
            return "polynomial"
        elif avg_ratio < 6.0:
            return "exponential"
        else:
            return "super_exponential"

    def _probe_decay(self, expr, variable: str) -> tuple:
        v = Symbol(variable)
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if inner.has(v):
                    # Check for Gaussian pattern: exp(-c*x^2)
                    x2_terms = [t for t in inner.atoms() if hasattr(t, "is_Pow") and getattr(t, "is_Pow", False)]
                    for t in x2_terms:
                        try:
                            if t.base == v and t.exp == 2:
                                coeff = float(t.as_coeff_Mul()[0])
                                if coeff < 0:
                                    return ("gaussian", abs(coeff))
                        except Exception:
                            pass
                    # Also check -(x**2) pattern directly
                    neg_x2 = -v ** 2
                    if inner == neg_x2 or inner.expand().has(neg_x2):
                        return ("gaussian", 1.0)
            except Exception:
                pass
        for sub in expr.atoms(exp):
            try:
                inner = sub.args[0]
                if inner == -v or (hasattr(inner, "is_Mul") and inner.is_Mul and any(t == -v for t in inner.args)):
                    return ("exponential", 1.0)
            except Exception:
                pass
        try:
            den = expr.as_numer_denom()[1]
            if den.has(v):
                power_terms = den.atoms()
                for pt in power_terms:
                    if hasattr(pt, "exp") and isinstance(pt.exp, (int, float)) and pt.exp < 0:
                        return ("algebraic", float(-pt.exp))
        except Exception:
            pass
        if expr.has(sin) or expr.has(cos):
            return ("oscillatory", 0.0)
        return ("none", 0.0)

    def _probe_periodicity(self, expr, a: float, b: float) -> tuple:
        v = self._var
        n_samples = 20
        xs = np.linspace(a, b, n_samples)
        ys = []
        for x in xs:
            try:
                val = float(N(expr.subs(v, x), 10))
                if math.isfinite(val):
                    ys.append(val)
            except Exception:
                pass
        if len(ys) < 6:
            return False, None
        try:
            fa = float(N(expr.subs(v, a), 10))
            fb = float(N(expr.subs(v, b), 10))
            df = diff(expr, v)
            fa_prime = float(N(df.subs(v, a), 10)) if math.isfinite(fa) else None
            fb_prime = float(N(df.subs(v, b), 10)) if math.isfinite(fb) else None
            if (math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) < 1e-6):
                if fa_prime is not None and fb_prime is not None:
                    if abs(fa_prime - fb_prime) < 1e-4:
                        return True, b - a
        except Exception:
            pass
        if expr.has(sin) or expr.has(cos):
            period = None
            for sub in list(expr.atoms(sin)) + list(expr.atoms(cos)):
                try:
                    arg = sub.args[0]
                    coeff = arg.as_coeff_mul(v)[0] if hasattr(arg, "as_coeff_mul") else S.One
                    cf = float(coeff)
                    if math.isfinite(cf) and abs(cf) > 1e-12:
                        period = 2 * float(pi) / abs(cf)
                except Exception:
                    pass
            if period is not None and (b - a) <= period + 1e-6:
                return True, period
        return False, None

    # ------------------------------------------------------------------ #
    #  Recommendation engine                                              #
    # ------------------------------------------------------------------ #

    @dataclass
    class _Rec:
        family: PolynomialFamily
        confidence: str
        reason: str

    def _recommend_family(self, analysis: FunctionAnalysis, a: float, b: float) -> "QuadratureAnalyzer._Rec":
        iv = analysis.interval_type
        has_end_sg = analysis.has_endpoint_singularity
        has_int_sg = analysis.has_interior_singularity
        decay = analysis.decay_type
        is_periodic = analysis.is_periodic_on_interval

        if iv == "infinite":
            if decay in ("gaussian",):
                return self._Rec(PolynomialFamily.HERMITE, "high",
                    "Infinite domain with Gaussian-weighted integrand. Gauss-Hermite quadrature absorbs the e^{-x^2} weight, yielding machine-precision results with O(sqrt(desired_digits)) nodes.")
            elif decay in ("exponential",):
                return self._Rec(PolynomialFamily.HERMITE, "medium",
                    "Infinite domain with exponential decay. Gauss-Hermite can still be effective if the decay is fast enough; otherwise consider a variable transformation to a finite interval and use Legendre.")
            else:
                return self._Rec(PolynomialFamily.HERMITE, "low",
                    "Infinite domain with unknown or slow decay. Gauss-Hermite may converge slowly; consider transforming the integral to a finite interval first.")

        if iv == "semi_infinite":
            if decay in ("exponential",):
                return self._Rec(PolynomialFamily.LAGUERRE, "high",
                    "Semi-infinite domain with exponential decay. Gauss-Laguerre absorbs the e^{-x} weight, ideal for Laplace-type integrals.")
            elif decay in ("algebraic",):
                return self._Rec(PolynomialFamily.LAGUERRE, "medium",
                    "Semi-infinite domain with algebraic decay. Gauss-Laguerre is usable but may need higher degree; consider variable substitution t = x/(1+x) to map [0, inf) -> [0, 1) and use Legendre.")
            else:
                return self._Rec(PolynomialFamily.LAGUERRE, "medium",
                    "Semi-infinite domain. Gauss-Laguerre is the default choice; for slow decay consider mapping to a finite interval.")

        if iv == "finite":
            if has_end_sg:
                return self._Rec(PolynomialFamily.CHEBYSHEV, "high",
                    f"Endpoint singularity detected at {analysis.singularities}. Chebyshev nodes cluster near endpoints, and Clenshaw-Curtis quadrature handles endpoint singularities much better than Gauss-Legendre.")
            if has_int_sg:
                return self._Rec(PolynomialFamily.LEGENDRE, "low",
                    f"Interior singularity detected at {analysis.singularities}. Standard Gaussian quadrature will converge slowly. Consider splitting the integral at the singularity or applying a variable transformation to remove it.")
            if is_periodic:
                return self._Rec(PolynomialFamily.CHEBYSHEV, "high",
                    f"Function appears periodic on [{a}, {b}] with period ~{analysis.approximate_period}. Clenshaw-Curtis quadrature exploits periodicity via FFT and achieves spectral accuracy.")
            return self._Rec(PolynomialFamily.LEGENDRE, "high",
                "Smooth function on a finite interval with no singularities. Gauss-Legendre is the optimal choice: it integrates polynomials of degree up to 2n-1 exactly and converges exponentially for analytic integrands.")

        return self._Rec(PolynomialFamily.LEGENDRE, "low", "Could not determine optimal family; defaulting to Gauss-Legendre.")

    # ------------------------------------------------------------------ #
    #  Degree-selection criteria                                          #
    # ------------------------------------------------------------------ #

    def _degree_criteria(self, analysis: FunctionAnalysis) -> list:
        criteria: list = []
        fam = analysis.recommended_family
        iv = analysis.interval_type
        growth = analysis.derivative_growth_rate
        has_sg = analysis.has_endpoint_singularity or analysis.has_interior_singularity
        is_periodic = analysis.is_periodic_on_interval

        # General convergence principle
        criteria.append(
            "Gaussian quadrature with n nodes integrates exactly all polynomials of degree <= 2n-1. For non-polynomial integrands, the error decays as O(|f^{(2n)}(xi)|) for some xi in the interval.")

        # Smoothness-based criterion
        if growth == "bounded":
            criteria.append(
                "Derivatives are bounded on the interval => f is very smooth (likely analytic). Exponential convergence is expected; n=16-32 usually suffices for double-precision accuracy.")
        elif growth == "polynomial":
            criteria.append(
                "Derivatives grow polynomially (like k!) => f is C^infty but not analytic. Algebraic convergence O(n^{-p}) expected; use n=32-128 and monitor convergence by doubling n.")
        elif growth == "exponential":
            criteria.append(
                "Derivatives grow exponentially (like C^k * k!) => f has a nearby complex singularity. Convergence is algebraic; use n=64-256 and consider a variable transformation to push singularities further away.")
        else:
            criteria.append(
                "Derivatives grow super-exponentially => f has a strong singularity or is not smooth. Standard Gaussian quadrature will converge slowly; consider splitting the interval, using a weight function that matches the singularity, or applying a change of variables.")

        # Singularity-based criterion
        if has_sg:
            criteria.append(
                "Singularities present: " + str(analysis.singularities) + ". This degrades convergence from exponential to algebraic (O(n^{-alpha}) where alpha depends on the singularity strength). Use n=64-256 and prefer Chebyshev nodes which cluster near endpoints.")

        # Periodicity criterion
        if is_periodic:
            criteria.append(
                "Function is periodic on the interval (period ~" + str(analysis.approximate_period) + "). Clenshaw-Curtis quadrature achieves spectral accuracy; n=16-64 typically reaches machine precision.")

        # Infinite-interval criterion
        if iv == "infinite":
            criteria.append(
                "Infinite domain with decay type '" + analysis.decay_type + "'. The effective number of nodes needed scales as O(sqrt(-log(tol))) for Gaussian-weighted integrals. For double precision (tol ~ 1e-15), n=32-64 is usually sufficient with the matching weight function.")
        elif iv == "semi_infinite":
            criteria.append(
                "Semi-infinite domain with decay type '" + analysis.decay_type + "'. Gauss-Laguerre absorbs e^{-x}; for slower decays, n=64-128 may be needed.")

        # Family-specific criterion
        if fam == PolynomialFamily.LEGENDRE:
            criteria.append(
                "Gauss-Legendre: optimal for smooth integrands on finite intervals. Error bound ~ M_{2n} * (b-a)^{2n+1} / ((2n+1) * 4^n * (n!)^2), where M_{2n} = max|f^{(2n)}|.")
        elif fam == PolynomialFamily.CHEBYSHEV:
            criteria.append(
                "Gauss-Chebyshev / Clenshaw-Curtis: optimal when endpoint singularities are present or the function is periodic. Clenshaw-Curtis converges as O(n^{-r}) where r depends on smoothness; for analytic functions, convergence is exponential.")
        elif fam == PolynomialFamily.HERMITE:
            criteria.append(
                "Gauss-Hermite: optimal for integrals over (-inf, +inf) with e^{-x^2} weight. Error decays as O(M_{2n} / 4^n * n!). For f(x)=1 (pure Gaussian integral), n=8 already gives full double-precision accuracy.")
        else:
            criteria.append(
                "Gauss-Laguerre: optimal for [0, +inf) with e^{-x} weight. Error decays as O(M_{2n} / 4^n * n!). For f(x)=1 (pure Laguerre integral), n=8 gives full double-precision accuracy.")

        # Practical convergence check
        criteria.append(
            "Practical rule: compute the integral with n and 2n nodes; if |I_{2n} - I_n| < tol * (1 + |I_{2n}|), stop. Otherwise double n until convergence or until n exceeds the warning threshold (Hermite: n>=100, others: n>=200).")

        return criteria

    def _suggest_degree_range(self, analysis: FunctionAnalysis) -> tuple:
        growth = analysis.derivative_growth_rate
        has_sg = analysis.has_endpoint_singularity or analysis.has_interior_singularity
        iv = analysis.interval_type
        fam = analysis.recommended_family

        if growth == "bounded":
            lo, hi = 8, 32
        elif growth == "polynomial":
            lo, hi = 16, 64
        elif growth == "exponential":
            lo, hi = 32, 128
        else:
            lo, hi = 64, 256

        if has_sg:
            lo = max(lo, 32)
            hi = max(hi, 128)
        if iv == "infinite" and fam in (PolynomialFamily.HERMITE, PolynomialFamily.LAGUERRE):
            lo = min(lo, 16)
            hi = min(hi, 64)
        if analysis.is_periodic_on_interval:
            lo = max(8, lo // 2)
            hi = max(32, hi // 2)
        return (lo, hi)

    @staticmethod
    def _fmt_val(v) -> str:
        if v == float("inf"):
            return "float(chr(39)+chr(105)+chr(110)+chr(102)+chr(39))"
        if v == float("-inf"):
            return "float(chr(39)+chr(45)+chr(105)+chr(110)+chr(102)+chr(39))"
        return f"{v:.6g}"

    def recommend_usage(self, analysis: FunctionAnalysis) -> str:
        fam = analysis.recommended_family
        n_max = analysis.suggested_max_n
        var = analysis.variable
        a_val = self._fmt_val(analysis.singularities[0]["location"]) if analysis.singularities else "-1.0"

        if fam == PolynomialFamily.LEGENDRE:
            return (
                f"# Recommended: Gauss-Legendre on [{analysis.original_expr}]\n"
                f"from legendre import LegendreQuadrature\n"
                f"import numpy as np\n\n"
                f"quad = LegendreQuadrature(n={n_max}, use_mpmath=False)\n"
                f"result = quad.integrate_transformed(\n"
                f"    lambda {var}: {analysis.original_expr},\n"
                f"    a={a_val}, b=1.0,\n"
                f"    nodes=quad.nodes, weights=quad.weights\n"
                f")\n"
            )
        elif fam == PolynomialFamily.CHEBYSHEV:
            return (
                f"# Recommended: Gauss-Chebyshev or Clenshaw-Curtis\n"
                f"from chebyshev import ChebyshevQuadrature\n"
                f"import numpy as np\n\n"
                f"q = ChebyshevQuadrature()\n"
                f"result = q.gauss_chebyshev_quadrature(\n"
                f"    lambda {var}: {analysis.original_expr}, n={n_max}\n"
                f")\n"
            )
        elif fam == PolynomialFamily.HERMITE:
            return (
                f"# Recommended: Gauss-Hermite on (-inf, +inf)\n"
                f"from hermite import GaussHermiteQuadrature\n"
                f"import numpy as np\n\n"
                f"quad = GaussHermiteQuadrature(n={n_max}, use_mpmath=True)\n"
                f"result = quad.integrate(lambda {var}: {analysis.original_expr})\n"
            )
        else:
            return (
                f"# Recommended: Gauss-Laguerre on [0, +inf)\n"
                f"from laguerre import LaguerreQuadrature\n"
                f"import numpy as np\n\n"
                f"quad = LaguerreQuadrature(n={n_max}, alpha=0.0, use_mpmath=False)\n"
                f"result = quad.integrate(lambda {var}: {analysis.original_expr})\n"
            )


# ---------------------------------------------------------------------------
#  Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = QuadratureAnalyzer()

    examples = [
        ("exp(-x**2) * sin(x)", (-1, 1), "Smooth analytic function on [-1,1]"),
        ("1 / sqrt(1 - x**2)", (-1, 1), "Endpoint singularity at +/-1"),
        ("exp(-x**2)", None, "Gaussian decay => infinite interval"),
        ("log(x) * exp(-x)", None, "Log singularity + exponential decay"),
        ("cos(3*x)", (-float(pi), float(pi)), "Periodic function on [-pi, pi]"),
        ("x**2 * exp(-x)", (0, float("inf")), "Laguerre-type integral"),
        ("1 / (1 + x**4)", (-float("inf"), float("inf")), "Algebraic decay on infinite interval"),
    ]

    for expr_str, interval, desc in examples:
        print(f"\n{'=' * 70}")
        print(f"  {desc}")
        print(f"  Expression: {expr_str}")
        print(f"  Interval:   {interval}")
        print(f"{'=' * 70}")

        analysis = analyzer.analyze(expr_str, interval=interval)

        print(f"\n  Recommended family : {analysis.recommended_family.value}")
        print(f"  Confidence         : {analysis.confidence}")
        reason_preview = analysis.recommendation_reason[:120]
        print(f"  Reason             : {reason_preview}...")
        print(f"  Suggested degree   : [{analysis.suggested_min_n}, {analysis.suggested_max_n}]")

        if analysis.singularities:
            print(f"  Singularities      : {analysis.singularities}")
        if analysis.is_periodic_on_interval:
            print(f"  Periodic           : True (period ~{analysis.approximate_period})")
        print(f"  Derivative growth  : {analysis.derivative_growth_rate}")

        print("\n  Degree-selection criteria:")
        for i, c in enumerate(analysis.degree_criteria, 1):
            words = c.split()
            line = ""
            for w in words:
                if len(line) + len(w) + 1 > 76:
                    print(f"    {line}")
                    line = f"      {w}"
                else:
                    prefix = "" if not line else " "
                    line = f"{line}{prefix}{w}"
            if line:
                print(f"    {line}")

        print("\n  Usage snippet:")
        usage = analyzer.recommend_usage(analysis)
        for uline in usage.strip().splitlines():
            print(f"    {uline}")

    print(f"\n{'=' * 70}")
    print("  Done.")
