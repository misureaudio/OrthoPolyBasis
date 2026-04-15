"""Hermite Polynomial Symbolic Module - The Source.

This module generates exact rational coefficients for Hermite polynomials using SymPy.
It serves as the ground truth factory that feeds both high_precision.py and numerical.py.

Three-Term Recurrence (Physicist's convention):
    H_{n+1}(x) = 2*x*H_n(x) - 2*n*H_{n-1}(x)
    with H_0(x) = 1, H_1(x) = 2*x
"""

from __future__ import annotations
from typing import Optional, Union, List
import sympy as sp
from sympy import S, Rational, Poly, Symbol
from sympy.core.expr import Expr


class HermiteSymbolic:
    """Exact symbolic representation of Hermite polynomials.
    
    Uses SymPy's arbitrary-precision rational arithmetic to maintain exactness
    throughout all operations. This is the authoritative source for coefficients.
    """
    
    def __init__(
        self,
        degree: int,
        convention: str = "physicist",
        x_symbol: Optional[Symbol] = None
    ):
        if degree < 0:
            raise ValueError(f"Degree must be non-negative, got {degree}")
        if convention not in ("physicist", "probabilist"):
            raise ValueError(f"Convention must be 'physicist' or 'probabilist'")
        
        self.degree = degree
        self.convention = convention
        self._x: Symbol = x_symbol if x_symbol is not None else Symbol("x")
        self._symbolic_poly: Expr = self._generate_via_recurrence()
    
    def _generate_via_recurrence(self) -> Expr:
        """Generate H_n(x) using Three-Term Recurrence for numerical stability."""
        # BEGIN GEMINI
        """
        # Align by prepending zeros to the shorter list
        max_len = max(len(two_x_h_curr), len(two_n_h_prev))
        two_x_h_curr = [mp.mpf(0)] * (max_len - len(two_x_h_curr)) + two_x_h_curr
        two_n_h_prev = [mp.mpf(0)] * (max_len - len(two_n_h_prev)) + two_n_h_prev

        h_next = [a - b for a, b in zip(two_x_h_curr, two_n_h_prev)]
        """
        # END GEMINI
        if self.degree == 0:
            return S.One
        if self.degree == 1:
            return 2 * self._x if self.convention == "physicist" else self._x
        
        h_prev: Expr = S.One
        h_curr: Expr = 2 * self._x if self.convention == "physicist" else self._x
        
        for n in range(1, self.degree):
            if self.convention == "physicist":
                h_next: Expr = 2 * self._x * h_curr - 2 * n * h_prev
            else:
                h_next: Expr = self._x * h_curr - n * h_prev
            
            h_prev, h_curr = h_curr, h_next
        
        return sp.expand(h_curr)
    
    def __repr__(self) -> str:
        conv = "H" if self.convention == "physicist" else "He"
        return f"HermiteSymbolic({conv}_{self.degree}, convention='{self.convention}')"
    
    def __str__(self) -> str:
        return str(self._symbolic_poly)
    
    def _to_expr(self, other: Union["HermiteSymbolic", Expr, int]) -> Expr:
        if isinstance(other, HermiteSymbolic):
            return other._symbolic_poly
        return S(other)
    
    def __add__(self, other) -> "HermiteSymbolic":
        result = sp.expand(self._symbolic_poly + self._to_expr(other))
        return HermiteSymbolic.from_expression(result, self.convention, self._x)
    
    def __radd__(self, other) -> "HermiteSymbolic":
        return self.__add__(other)
    
    def __sub__(self, other) -> "HermiteSymbolic":
        result = sp.expand(self._symbolic_poly - self._to_expr(other))
        return HermiteSymbolic.from_expression(result, self.convention, self._x)
    
    def __rsub__(self, other) -> "HermiteSymbolic":
        result = sp.expand(self._to_expr(other) - self._symbolic_poly)
        return HermiteSymbolic.from_expression(result, self.convention, self._x)
    
    def __mul__(self, other) -> "HermiteSymbolic":
        result = sp.expand(self._symbolic_poly * self._to_expr(other))
        return HermiteSymbolic.from_expression(result, self.convention, self._x)
    
    def __rmul__(self, other) -> "HermiteSymbolic":
        return self.__mul__(other)
    
    def __neg__(self) -> "HermiteSymbolic":
        return HermiteSymbolic.from_expression(-self._symbolic_poly, self.convention, self._x)
    
    def derivative(self, order: int = 1) -> "HermiteSymbolic":
        if order < 0:
            raise ValueError("Derivative order must be non-negative")
        if order == 0:
            return self
        
        if self.convention == "physicist" and order <= self.degree:
            coeff = pow(S(2), order) * sp.factorial(self.degree) / sp.factorial(self.degree - order)
            if self.degree == order:
                return HermiteSymbolic.from_expression(coeff, self.convention, self._x)
            result = coeff * HermiteSymbolic(self.degree - order, self.convention, self._x)._symbolic_poly
        else:
            result = sp.diff(self._symbolic_poly, self._x, order)
        
        return HermiteSymbolic.from_expression(result, self.convention, self._x)
    
    def indefinite_integral(self) -> "HermiteSymbolic":
        if self.convention == "physicist":
            result = -HermiteSymbolic(self.degree + 1, self.convention, self._x)._symbolic_poly / (2 * (self.degree + 1))
        else:
            result = -HermiteSymbolic(self.degree + 1, self.convention, self._x)._symbolic_poly / (self.degree + 1)
        return HermiteSymbolic.from_expression(result, self.convention, self._x)
    
    def get_coefficients(self, ascending: bool = False) -> List[Rational]:
        poly = Poly(self._symbolic_poly, self._x)
        coeffs = poly.all_coeffs()
        if ascending:
            coeffs = list(reversed(coeffs))
        return [c if isinstance(c, Rational) else Rational(c) for c in coeffs]
    
    def to_numerical_coeffs(self) -> List[float]:
        """Export coefficients as Python floats for numerical.py."""
        coeffs = self.get_coefficients(ascending=False)
        return [float(c.evalf()) for c in coeffs]
    
    def to_high_precision_coeffs(self, dps: int = 50) -> List[str]:
        """Export coefficients as strings for exact mpmath reconstruction."""
        coeffs = self.get_coefficients(ascending=False)
        return [str(c.evalf(dps + 10)) for c in coeffs]
    
    def to_dict(self) -> dict:
        poly = Poly(self._symbolic_poly, self._x)
        asm_dict = poly.as_dict()
        return {int(p): Rational(coeff) for p, coeff in asm_dict.items()}
    
    @classmethod
    def from_expression(cls, expr: Expr, convention: str = "physicist", x_symbol: Optional[Symbol] = None) -> "HermiteSymbolic":
        instance = object.__new__(cls)
        poly_expr = sp.expand(expr)
        poly_obj = Poly(poly_expr, x_symbol if x_symbol else Symbol("x"))
        instance.convention = convention
        instance._x = x_symbol if x_symbol else Symbol("x")
        instance._symbolic_poly = poly_expr
        instance.degree = poly_obj.degree()
        return instance
    
    @classmethod
    def from_coefficients(cls, coefficients: List[Union[int, float, Rational, Expr]], convention: str = "physicist", ascending: bool = False, x_symbol: Optional[Symbol] = None) -> "HermiteSymbolic":
        x = x_symbol if x_symbol else Symbol("x")
        coeffs = [S(c) for c in coefficients]
        if ascending:
            expr = sum(c * x**i for i, c in enumerate(coeffs))
        else:
            expr = sum(c * x**(len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
        return cls.from_expression(expr, convention, x)
    
    def evaluate_symbolic(self, point: Union[int, float, Rational, Expr]) -> Expr:
        return self._symbolic_poly.subs(self._x, S(point))
    
    def get_leading_coefficient(self) -> Rational:
        coeffs = self.get_coefficients(ascending=False)
        return coeffs[0] if coeffs else S.Zero
    
    def is_even(self) -> bool:
        return self.degree % 2 == 0
    
    def is_odd(self) -> bool:
        return self.degree % 2 == 1


def hermite_symbolic_basis(n_max: int, convention: str = "physicist") -> List[HermiteSymbolic]:
    """Generate complete Hermite basis {H_0, H_1, ..., H_n}."""
    return [HermiteSymbolic(n, convention) for n in range(n_max + 1)]
