# Sympy integration for Legendre polynomials
from typing import List, Optional
try:
    from sympy import symbols, Symbol, Function, diff, integrate, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

def _validate_n(n: int) -> None:
    """Validate that n is a non-negative integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

class SympyLegendreGenerator:
    """Generate Legendre polynomials using SymPy for symbolic computation."""
    
    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for SympyLegendreGenerator")
        self.x = symbols("x")
        self._cache = {}
    
    def get_polynomial(self, n: int) -> Function:
        """Get the nth Legendre polynomial as a SymPy expression."""
        _validate_n(n)
        if n in self._cache:
            return self._cache[n]
        from sympy.polys.orthopolys import legendre_poly
        poly = legendre_poly(n, self.x)
        self._cache[n] = poly
        return poly
    
    def get_polynomial_manual(self, n: int) -> Function:
        """Generate Legendre polynomial manually using recurrence."""
        _validate_n(n)
        if n in self._cache:
            return self._cache[n]
        if n == 0:
            poly = 1
        elif n == 1:
            poly = self.x
        else:
            prev2 = 1
            prev1 = self.x
            for k in range(2, n + 1):
                poly = ((2*k - 1) * self.x * prev1 - (k - 1) * prev2) / k
                prev2 = prev1
                prev1 = poly
        self._cache[n] = poly
        return poly
    
    def get_derivative(self, n: int) -> Function:
        """Get the derivative of the nth Legendre polynomial."""
        _validate_n(n)
        poly = self.get_polynomial(n)
        return diff(poly, self.x)
    
    def get_integral(self, n: int) -> Function:
        """Get the indefinite integral of the nth Legendre polynomial."""
        _validate_n(n)
        poly = self.get_polynomial(n)
        return integrate(poly, self.x)
    
    def evaluate(self, x_val, n: int) -> float:
        """Evaluate P_n(x) at a specific point."""
        _validate_n(n)
        poly = self.get_polynomial(n)
        return float(poly.subs(self.x, x_val))
    
    def generate_basis(self, max_n: int) -> List[Function]:
        """Generate basis from P_0 to P_max_n."""
        _validate_n(max_n)
        return [self.get_polynomial(n) for n in range(max_n + 1)]


def generate_sympy_legendre(n: int, use_manual: bool = False) -> Function:
    """Convenience function to get the nth Legendre polynomial."""
    _validate_n(n)
    gen = SympyLegendreGenerator()
    if use_manual:
        return gen.get_polynomial_manual(n)
    return gen.get_polynomial(n)


def get_sympy_legendre_basis(max_n: int) -> List[Function]:
    """Convenience function to generate a basis of Legendre polynomials."""
    _validate_n(max_n)
    gen = SympyLegendreGenerator()
    return gen.generate_basis(max_n)
