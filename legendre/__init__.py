from .core import (
    legendre_polynomial,
    legendre_coefficients,
    legendre_coefficients_descending,
    legendre_coefficients_ascending,
    legendre_derivative,
    legendre_integral,
)
from .sympy_integration import generate_sympy_legendre, get_sympy_legendre_basis
from .numpy_integration import (
    generate_numpy_legendre,
    generate_numpy_legendre_descending,
    generate_numpy_legendre_ascending,
    get_numpy_legendre_basis,
    get_numpy_legendre_basis_descending,
    get_numpy_legendre_basis_ascending,
    evaluate_numpy_legendre,
)
from .mpmath_integration import (
    generate_mpmath_legendre,
    get_mpmath_legendre_basis,
    get_mpmath_legendre_coefficients,
    get_mpmath_legendre_coefficients_descending,
    get_mpmath_legendre_coefficients_ascending,
    evaluate_mpmath_legendre,
)
from .quadrature import (
    GaussLegendreQuadrature,
    gauss_legendre,
    gauss_legendre_newton,
    gauss_legendre_golub_welsch,
)
from .quadrature_high_precision import (
    HighPrecisionGaussLegendre,
    gauss_legendre_high_precision,
)

__all__ = [
    # Core functions (evaluation)
    "legendre_polynomial",
    
    # Core coefficient functions (deprecated ambiguous names)
    "legendre_coefficients",
    "legendre_derivative",
    "legendre_integral",
    
    # Core coefficient functions (explicit naming - preferred)
    "legendre_coefficients_descending",
    "legendre_coefficients_ascending",
    
    # Sympy integration
    "generate_sympy_legendre",
    "get_sympy_legendre_basis",
    
    # Numpy integration (deprecated ambiguous names)
    "generate_numpy_legendre",
    "get_numpy_legendre_basis",
    "evaluate_numpy_legendre",
    
    # Numpy integration (explicit naming - preferred)
    "generate_numpy_legendre_descending",
    "generate_numpy_legendre_ascending",
    "get_numpy_legendre_basis_descending",
    "get_numpy_legendre_basis_ascending",
    
    # Mpmath integration
    "generate_mpmath_legendre",
    "get_mpmath_legendre_basis",
    "evaluate_mpmath_legendre",
    
    # Mpmath coefficient functions (deprecated ambiguous name)
    "get_mpmath_legendre_coefficients",
    
    # Mpmath coefficient functions (explicit naming - preferred)
    "get_mpmath_legendre_coefficients_descending",
    "get_mpmath_legendre_coefficients_ascending",
    
    # Gauss-Legendre Quadrature
    "GaussLegendreQuadrature",
    "gauss_legendre",
    "gauss_legendre_newton",
    "gauss_legendre_golub_welsch",
    
    # High-precision quadrature
    "HighPrecisionGaussLegendre",
    "gauss_legendre_high_precision",
]

__version__ = "1.0.0"
