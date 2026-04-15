# Sympy integration for Chebyshev polynomials
from typing import List, Optional
try:
    from sympy import symbols, Symbol, Function, diff, integrate, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

class SympyChebyshevGenerator:
    """Generate Chebyshev polynomials using SymPy for symbolic computation."""
    
    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for SympyChebyshevGenerator")
        self.x = symbols('x')
        self._cache = {}
    
    def get_polynomial(self, n: int) -> Function:
        """Get the nth Chebyshev polynomial as a SymPy expression."""
        if n in self._cache:
            return self._cache[n]
        from sympy.polys.orthopolys import chebyshevt_poly
        poly = chebyshevt_poly(n, self.x)
        self._cache[n] = poly
        return poly
    
    def get_polynomial_manual(self, n: int) -> Function:
        """Generate Chebyshev polynomial manually using recurrence."""
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
                # T_k(x) = 2*x*T_{k-1}(x) - T_{k-2}(x)
                poly = 2 * self.x * prev1 - prev2
                prev2 = prev1
                prev1 = poly
        self._cache[n] = poly
        return poly
    
    def get_derivative(self, n: int) -> Function:
        """Get the derivative of the nth Chebyshev polynomial."""
        poly = self.get_polynomial(n)
        return diff(poly, self.x)
    
    def get_integral(self, n: int) -> Function:
        """Get the indefinite integral of the nth Chebyshev polynomial."""
        poly = self.get_polynomial(n)
        return integrate(poly, self.x)
    
    def evaluate(self, x_val, n: int) -> float:
        """Evaluate T_n(x) at a specific point."""
        poly = self.get_polynomial(n)
        return float(poly.subs(self.x, x_val))
    
    def generate_basis(self, max_n: int) -> List[Function]:
        """Generate basis from T_0 to T_max_n."""
        return [self.get_polynomial(n) for n in range(max_n + 1)]


def generate_sympy_chebyshev(n: int, use_manual: bool = False) -> Function:
    """Convenience function to get the nth Chebyshev polynomial."""
    gen = SympyChebyshevGenerator()
    if use_manual:
        return gen.get_polynomial_manual(n)
    return gen.get_polynomial(n)


def get_sympy_chebyshev_basis(max_n: int) -> List[Function]:
    """Convenience function to generate a basis of Chebyshev polynomials."""
    gen = SympyChebyshevGenerator()
    return gen.generate_basis(max_n)