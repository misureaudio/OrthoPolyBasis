from typing import List, Optional, Callable
try:
    from mpmath import mp, chebyt as mpmath_chebyt, diff, quad
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

class MpmathChebyshevGenerator:
    def __init__(self, default_dps: int = 50):
        if not MPMATH_AVAILABLE:
            raise ImportError("mpmath is required for MpmathChebyshevGenerator")
        self._cache = {}
        self.default_dps = default_dps
    
    def get_polynomial(self, n: int) -> Callable:
        if n in self._cache:
            return self._cache[n]
        
        # We wrap the call so it respects the precision of the caller
        def poly(x):
            # No internal dps setting; let the user's context decide
            return mpmath_chebyt(n, x)
        
        self._cache[n] = poly
        return poly
    
    def get_derivative(self, n: int) -> Callable:
        def deriv(x):
            # We convert x to mpf within the current precision context
            x_mp = mp.mpf(str(x))
            return diff(lambda t: mpmath_chebyt(n, t), x_mp)
        return deriv
    
    def get_integral(self, n: int) -> Callable:
        def integ(x):
            x_mp = mp.mpf(str(x))
            if n == 0:
                return x_mp
            else:
                # quad is very sensitive to dps. 
                # It will use whatever mp.dps is currently set to.
                return quad(lambda t: mpmath_chebyt(n, t), [0, x_mp])
        return integ
    
    def evaluate(self, x: str, n: int, dps: Optional[int] = None) -> mp.mpf:
        """Evaluate T_n(x) without polluting global state."""
        target_dps = dps or self.default_dps
        # Use workdps to ensure precision is restored after the call
        with mp.workdps(target_dps):
            poly = self.get_polynomial(n)
            return poly(mp.mpf(str(x)))

    def evaluate_batch(self, xs: List[str], n: int, dps: Optional[int] = None) -> List[mp.mpf]:
        target_dps = dps or self.default_dps
        with mp.workdps(target_dps):
            poly = self.get_polynomial(n)
            return [poly(mp.mpf(str(x))) for x in xs]


# --- Convenience Functions ---

def evaluate_mpmath_chebyshev(x: str, n: int, dps: int = 50) -> mp.mpf:
    gen = MpmathChebyshevGenerator(default_dps=dps)
    return gen.evaluate(x, n)

# Note: The generate functions should return functions that 
# the user can then call inside their own `with mp.workdps():` blocks.
def generate_mpmath_chebyshev(n: int) -> Callable:
    return MpmathChebyshevGenerator().get_polynomial(n)


def generate_basis(self, max_n: int) -> List[Callable]:
    """Generate basis from T_0 to T_max_n."""
    return [self.get_polynomial(n) for n in range(max_n + 1)]


def get_mpmath_chebyshev_basis(max_n: int) -> List[Callable]:
    """Convenience function to generate a basis of Chebyshev polynomials."""
    gen = MpmathChebyshevGenerator()
    return gen.generate_basis(max_n)