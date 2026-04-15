import sys
sys.path.insert(0, '.')
from .quadrature_roots import get_chebyshev_zeros, get_chebyshev_extrema
print("Zeros T_4:", get_chebyshev_zeros(4))
print("Extrema T_3:", get_chebyshev_extrema(3))
