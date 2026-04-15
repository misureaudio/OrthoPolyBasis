# Mpmath integration for Legendre polynomials
from typing import List, Optional
try:
    from mpmath import mp, mpc, legendre as mpmath_legendre, diff, quad
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

def _validate_n(n: int) -> None:
    """Validate that n is a non-negative integer."""
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

class MpmathLegendreGenerator:
    """Generate Legendre polynomials using mpmath for arbitrary precision.
    
    COEFFICIENT ORDER CONVENTION:
    This module uses DESCENDING order (highest degree first) by default,
    matching the core module and traditional mathematical convention:
        P_n(x) = c_0*x^n + c_1*x^(n-1) + ... + c_n
    
    Use get_coefficients_ascending() for ascending order convention.
    """
    
    def __init__(self):
        if not MPMATH_AVAILABLE:
            raise ImportError("mpmath is required for MpmathLegendreGenerator")
        self._cache = {}
        self._coeff_cache_desc = {}
        self._coeff_cache_asc = {}
    
    def get_polynomial(self, n: int) -> callable:
        """Get the nth Legendre polynomial as a callable function.
        Uses mpmath's built-in legendre function with arbitrary precision.
        """
        _validate_n(n)
        if n in self._cache:
            return self._cache[n]
        
        def poly(x):
            return mpmath_legendre(n, x)
        
        self._cache[n] = poly
        return poly
    
    def get_polynomial_manual(self, n: int) -> callable:
        """Generate Legendre polynomial manually using recurrence with arbitrary precision.
        
        Uses the current mpmath precision setting (mp.dps). Does not modify global state.
        """
        _validate_n(n)
        if n in self._cache:
            return self._cache[n]
        
        def poly(x):
            x_mp = mp.mpf(str(x))
            
            if n == 0:
                return mp.mpf("1")
            elif n == 1:
                return x_mp
            else:
                prev2 = mp.mpf("1")
                prev1 = x_mp
                for k in range(2, n + 1):
                    # (k)P_k(x) = (2k-1)x*P_{k-1}(x) - (k-1)*P_{k-2}(x)
                    poly_val = ((2*k - 1) * x_mp * prev1 - (k - 1) * prev2) / k
                    prev2 = prev1
                    prev1 = poly_val
                return prev1
        
        self._cache[n] = poly
        return poly
    
    def get_coefficients_descending(self, n: int) -> List[mp.mpf]:
        """Get coefficients in DESCENDING order (highest degree first).
        
        This is the DEFAULT convention matching core.py and traditional math.
        P_n(x) = c_0*x^n + c_1*x^(n-1) + ... + c_n
        
        Uses arbitrary precision arithmetic. Coefficients are mp.mpf objects.
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            List of mp.mpf coefficients [c_0, c_1, ..., c_n]
            
        Example:
            >>> gen.get_coefficients_descending(2)
            [mpf('1.5'), mpf('0.0'), mpf('-0.5')]  # Represents 1.5*x^2 - 0.5
        """
        _validate_n(n)
        if n in self._coeff_cache_desc:
            return self._coeff_cache_desc[n].copy()
        
        if n == 0:
            coeffs = [mp.mpf("1")]
        elif n == 1:
            coeffs = [mp.mpf("1"), mp.mpf("0")]
        else:
            # Use list indexed by power for intermediate calculations (ascending)
            # prev2 represents P_{k-2}, prev1 represents P_{k-1}
            prev2 = [mp.mpf("1")]  # P_0 = 1
            prev1 = [mp.mpf("0"), mp.mpf("1")]  # P_1 = x
            
            for k in range(2, n + 1):
                # (k)P_k = (2k-1)x*P_{k-1} - (k-1)*P_{k-2}
                current = [mp.mpf("0")] * (k + 1)
                
                # (2k-1)x * P_{k-1}: shift by 1 and scale
                factor1 = mp.mpf(2 * k - 1)
                for i, coeff in enumerate(prev1):
                    current[i + 1] = factor1 * coeff
                
                # -(k-1) * P_{k-2}
                factor2 = mp.mpf(k - 1)
                for i, coeff in enumerate(prev2):
                    current[i] -= factor2 * coeff
                
                # Divide all by k
                for i in range(len(current)):
                    current[i] /= k
                
                prev2 = prev1
                prev1 = current
            
            # Convert to descending order
            coeffs = prev1[::-1]
        
        self._coeff_cache_desc[n] = coeffs
        return coeffs.copy()
    
    def get_coefficients_ascending(self, n: int) -> List[mp.mpf]:
        """Get coefficients in ASCENDING order (constant term first).
        
        P_n(x) = c_0 + c_1*x + ... + c_n*x^n
        
        Uses arbitrary precision arithmetic. Coefficients are mp.mpf objects.
        
        Args:
            n: Degree of Legendre polynomial (non-negative integer)
            
        Returns:
            List of mp.mpf coefficients [c_0, c_1, ..., c_n]
            
        Example:
            >>> gen.get_coefficients_ascending(2)
            [mpf('-0.5'), mpf('0.0'), mpf('1.5')]  # Represents -0.5 + 1.5*x^2
        """
        _validate_n(n)
        if n in self._coeff_cache_asc:
            return self._coeff_cache_asc[n].copy()
        
        # Get descending and reverse
        desc_coeffs = self.get_coefficients_descending(n)
        asc_coeffs = desc_coeffs[::-1]
        
        self._coeff_cache_asc[n] = asc_coeffs
        return asc_coeffs.copy()
    
    def get_coefficients(self, n: int) -> List[mp.mpf]:
        """Get coefficients in DESCENDING order (default convention).
        
        Alias for get_coefficients_descending().
        
        DEPRECATED: Use get_coefficients_descending() or 
        get_coefficients_ascending() explicitly to avoid confusion.
        """
        import warnings
        warnings.warn(
            "get_coefficients() is deprecated. Use get_coefficients_descending() "
            "or get_coefficients_ascending() explicitly to avoid confusion about ordering.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_coefficients_descending(n)
    
    def get_derivative(self, n: int) -> callable:
        """Get the derivative of the nth Legendre polynomial.
        Uses the current mpmath precision setting (mp.dps).
        """
        _validate_n(n)
        poly = self.get_polynomial(n)
        
        def deriv(x):
            x_mp = mp.mpf(str(x))
            return diff(lambda t: mpmath_legendre(n, t), x_mp)
        
        return deriv
    
    def get_integral(self, n: int) -> callable:
        """Get the indefinite integral of the nth Legendre polynomial.
        Uses the relation: ?P_n(x)dx = (P_{n+1}(x) - P_{n-1}(x)) / (2n + 1)
        Uses the current mpmath precision setting (mp.dps).
        """
        _validate_n(n)
        
        def integ(x):
            x_mp = mp.mpf(str(x))
            if n == 0:
                return x_mp  # ?P_0(x)dx = ?1 dx = x
            else:
                return (mpmath_legendre(n + 1, x_mp) - mpmath_legendre(n - 1, x_mp)) / (2*n + 1)
        
        return integ
    
    def evaluate(self, x: str, n: int, dps: Optional[int] = None) -> mp.mpf:
        """Evaluate P_n(x) at point x with arbitrary precision.
        x should be a string representation for best precision.
        
        Args:
            x: Point to evaluate at (string for best precision)
            n: Degree of Legendre polynomial
            dps: Optional decimal places. If None, uses current mp.dps setting.
                To temporarily set precision without modifying global state,
                use: with mp.workdps(new_dps): gen.evaluate(...)
        """
        _validate_n(n)
        if dps is not None:
            mp.dps = dps
        poly = self.get_polynomial(n)
        return poly(mp.mpf(str(x)))
    
    def evaluate_batch(self, xs: List[str], n: int, dps: Optional[int] = None) -> List[mp.mpf]:
        """Evaluate P_n(x) at multiple points with arbitrary precision.
        
        Args:
            xs: Points to evaluate at (strings for best precision)
            n: Degree of Legendre polynomial
            dps: Optional decimal places. If None, uses current mp.dps setting.
        """
        _validate_n(n)
        if dps is not None:
            mp.dps = dps
        poly = self.get_polynomial(n)
        return [poly(mp.mpf(str(x))) for x in xs]
    
    def generate_basis(self, max_n: int) -> List[callable]:
        """Generate basis from P_0 to P_max_n as callable functions."""
        _validate_n(max_n)
        return [self.get_polynomial(n) for n in range(max_n + 1)]
    
    def generate_coefficient_basis(self, max_n: int) -> List[List[mp.mpf]]:
        """Generate basis coefficients from P_0 to P_max_n.
        
        Returns coefficients in DESCENDING order (default convention).
        
        DEPRECATED: Use generate_coefficient_basis_descending() or
        generate_coefficient_basis_ascending() explicitly.
        """
        import warnings
        warnings.warn(
            "generate_coefficient_basis() is deprecated. Use "
            "generate_coefficient_basis_descending() or generate_coefficient_basis_ascending() "
            "explicitly to avoid confusion about ordering.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.generate_coefficient_basis_descending(max_n)
    
    def generate_coefficient_basis_descending(self, max_n: int) -> List[List[mp.mpf]]:
        """Generate basis coefficients in descending order."""
        _validate_n(max_n)
        return [self.get_coefficients_descending(n) for n in range(max_n + 1)]
    
    def generate_coefficient_basis_ascending(self, max_n: int) -> List[List[mp.mpf]]:
        """Generate basis coefficients in ascending order."""
        _validate_n(max_n)
        return [self.get_coefficients_ascending(n) for n in range(max_n + 1)]


def generate_mpmath_legendre(n: int, dps: Optional[int] = None) -> callable:
    """Convenience function to get the nth Legendre polynomial as a callable.
    Returns a function that can be called with x values.
    
    Args:
        n: Degree of Legendre polynomial
        dps: Optional decimal places. If None, uses current mp.dps setting.
    """
    _validate_n(n)
    gen = MpmathLegendreGenerator()
    if dps is not None:
        mp.dps = dps
    return gen.get_polynomial(n)


def get_mpmath_legendre_basis(max_n: int, dps: Optional[int] = None) -> List[callable]:
    """Convenience function to generate a basis of Legendre polynomials.
    Returns list of callable functions with arbitrary precision.
    
    Args:
        max_n: Maximum degree
        dps: Optional decimal places. If None, uses current mp.dps setting.
    """
    _validate_n(max_n)
    gen = MpmathLegendreGenerator()
    if dps is not None:
        mp.dps = dps
    return gen.generate_basis(max_n)


def evaluate_mpmath_legendre(x: str, n: int, dps: Optional[int] = None) -> mp.mpf:
    """Convenience function to evaluate P_n(x) with arbitrary precision.
    
    Args:
        x: Point to evaluate at (string for best precision)
        n: Degree of Legendre polynomial
        dps: Optional decimal places. If None, uses current mp.dps setting.
    """
    _validate_n(n)
    gen = MpmathLegendreGenerator()
    return gen.evaluate(x, n, dps)


def get_mpmath_legendre_coefficients(n: int) -> List[mp.mpf]:
    """Convenience function to get coefficients in descending order.
    
    DEPRECATED: Use get_mpmath_legendre_coefficients_descending() or
    get_mpmath_legendre_coefficients_ascending() explicitly.
    """
    import warnings
    warnings.warn(
        "get_mpmath_legendre_coefficients() is deprecated. Use "
        "get_mpmath_legendre_coefficients_descending() or "
        "get_mpmath_legendre_coefficients_ascending() explicitly.",
        DeprecationWarning,
        stacklevel=2
    )
    _validate_n(n)
    gen = MpmathLegendreGenerator()
    return gen.get_coefficients_descending(n)


def get_mpmath_legendre_coefficients_descending(n: int) -> List[mp.mpf]:
    """Get coefficients in descending order (default convention)."""
    _validate_n(n)
    gen = MpmathLegendreGenerator()
    return gen.get_coefficients_descending(n)


def get_mpmath_legendre_coefficients_ascending(n: int) -> List[mp.mpf]:
    """Get coefficients in ascending order."""
    _validate_n(n)
    gen = MpmathLegendreGenerator()
    return gen.get_coefficients_ascending(n)
