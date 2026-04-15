"""Hermite Polynomial Basis Module - Flexible Layered Stack.

This package provides a three-layer architecture for Hermite polynomials:

    symbolic.py      (The Source)   Exact rational coefficients via SymPy
                                    Generates ground truth for all layers
    
    high_precision.py  (The Bridge)  50+ decimal precision via mpmath
                                    Critical for roots/weights when n > 100
    
    numerical.py     (The Engine)   Fast NumPy/SciPy array operations
                                    Optimized for bulk data processing
    
    integration.py   (The Client)   Quadrature and projection utilities
                                    Orchestrates the three layers
"""

from .symbolic import (
    HermiteSymbolic,
    hermite_symbolic_basis,
)

from .high_precision import (
    HermiteMPMath,
    hermite_high_precision_basis,
)

from .numerical import (
    HermitePolynomial,
    hermite_numerical_basis,
)

from .integration import (
    GaussHermiteQuadrature,
    HermiteProjection,
    hermite_transform,
    inverse_hermite_transform,
)

__all__ = [
    "HermiteSymbolic",
    "hermite_symbolic_basis",
    "HermiteMPMath",
    "hermite_high_precision_basis",
    "HermitePolynomial",
    "hermite_numerical_basis",
    "GaussHermiteQuadrature",
    "HermiteProjection",
    "hermite_transform",
    "inverse_hermite_transform",
]

__version__ = "1.0.0"