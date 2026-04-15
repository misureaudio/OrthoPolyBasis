# Chebyshev polynomial package initialization

from .core import (
    chebyshev_coefficients,
    chebyshev_derivative,
    chebyshev_integral,
)
from .sympy_integration import generate_sympy_chebyshev, get_sympy_chebyshev_basis
from .numpy_integration import generate_numpy_chebyshev, get_numpy_chebyshev_basis
from .mpmath_integration import generate_mpmath_chebyshev, get_mpmath_chebyshev_basis
from .quadrature_roots import (
    ChebyshevQuadrature,
    get_chebyshev_zeros,
    get_chebyshev_extrema,
    gauss_chebyshev_integrate,
    clenshaw_curtis_integrate,
)
from .clencurt import clencurt, clencurt_quadrature

__all__ = [
    # Core functions
    "chebyshev_coefficients",
    "chebyshev_derivative",
    "chebyshev_integral",
    
    # SymPy integration
    "generate_sympy_chebyshev",
    "get_sympy_chebyshev_basis",
    
    # NumPy integration
    "generate_numpy_chebyshev",
    "get_numpy_chebyshev_basis",
    
    # mpmath integration
    "generate_mpmath_chebyshev",
    "get_mpmath_chebyshev_basis",
    
    # Quadrature and roots
    "ChebyshevQuadrature",
    "get_chebyshev_zeros",
    "get_chebyshev_extrema",
    "gauss_chebyshev_integrate",
    "clenshaw_curtis_integrate",
    
    # FFT-based Clenshaw-Curtis
    "clencurt",
    "clencurt_quadrature",
]

__version__ = "1.2.0"
