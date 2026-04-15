"""Hermite Polynomial High-Precision Module."""

from __future__ import annotations
from typing import Optional, Union, List, Tuple
import mpmath as mp
from mpmath import mpf, mpi, matrix

class HermiteMPMath:
    def __init__(self, degree: int = 0, convention: str = "physicist", coeffs: Optional[List] = None, dps: int = 50):
        mp.dps = dps
        self.degree = degree
        self.convention = convention
        self._dps = dps
        if coeffs is not None:
            self._coeffs = [mp.mpf(str(c)) for c in coeffs]
            self.degree = len(self._coeffs) - 1
        else:
            self._coeffs = self._generate_via_recurrence()
    
    def _generate_via_recurrence(self):
        if self.degree == 0:
            return [mp.mpf(1)]
        h_prev = [mp.mpf(1)]
        if self.convention == "physicist":
            h_curr = [mp.mpf(0), mp.mpf(2)]
        else:
            h_curr = [mp.mpf(0), mp.mpf(1)]
        if self.degree == 1:
            return list(reversed(h_curr))
        for k in range(1, self.degree):
            x_h_curr = [mp.mpf(0)] + h_curr
            if self.convention == "physicist":
                term1 = [mp.mpf(2) * c for c in x_h_curr]
                padded_prev = h_prev + [mp.mpf(0)] * (len(term1) - len(h_prev))
                term2 = [mp.mpf(2 * k) * c for c in padded_prev]
            else:
                term1 = x_h_curr
                padded_prev = h_prev + [mp.mpf(0)] * (len(term1) - len(h_prev))
                term2 = [mp.mpf(k) * c for c in padded_prev]
            h_next = [t1 - t2 for t1, t2 in zip(term1, term2)]
            while len(h_next) > 1 and h_next[-1] == mp.mpf(0):
                h_next.pop()
            h_prev, h_curr = h_curr, h_next
        return list(reversed(h_curr))
    
    def __repr__(self):
        conv = "H" if self.convention == "physicist" else "He"
        return "HermiteMPMath(" + conv + "_" + str(self.degree) + ", dps=" + str(self._dps) + ")"
    
    def get_coefficients(self, ascending: bool = False):
        if ascending:
            return list(reversed(self._coeffs))
        return self._coeffs.copy()
    
    def evaluate(self, x):
        x = mp.mpf(str(x))
        result = mp.mpf(0)
        for coeff in self._coeffs:
            result = result * x + coeff
        return result
    
    def derivative(self, order: int = 1):
        if order == 0:
            return HermiteMPMath.from_coefficients(self._coeffs, self.convention, self._dps)
        coeffs = self._coeffs.copy()
        for _ in range(order):
            new_coeffs = [coeff * mp.mpf(len(coeffs) - 1 - i) for i, coeff in enumerate(coeffs[:-1])]
            coeffs = new_coeffs
            if not coeffs:
                break
        return HermiteMPMath.from_coefficients(coeffs if coeffs else [mp.mpf(0)], self.convention, self._dps)

    """
    def get_roots(self):
        if self.degree == 0:
            return []
        n = self.degree
        tol_conv = mp.power(10, -(self._dps - 5))
        # Use a more lenient tolerance for duplicate detection
        tol_dup = mp.power(10, -(self._dps // 3))
        
        bound = mp.sqrt(2 * n + 1) * 1.5
        num_samples = max(1000, 4 * n)
        x_min, x_max = -bound, bound
        dx = (x_max - x_min) / num_samples
        
        roots = []
        prev_x = x_min
        prev_y = self.evaluate(prev_x)
        
        for i in range(1, num_samples + 1):
            x = x_min + i * dx
            y = self.evaluate(x)
            
            if prev_y == 0:
                root = prev_x
            elif y == 0 or prev_y * y < 0:
                a, b = prev_x, x
                fa, fb = prev_y, y
                for _ in range(100):
                    mid = (a + b) / 2
                    fm = self.evaluate(mid)
                    if abs(fm) < tol_conv or (b - a) / 2 < tol_conv:
                        break
                    if fa * fm <= 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                root = (a + b) / 2
            else:
                prev_x, prev_y = x, y
                continue
            
            # Check for duplicates with more lenient tolerance
            is_new = True
            for existing in roots:
                if abs(root - existing) < tol_dup:
                    is_new = False
                    break
            if is_new:
                roots.append(root)
            
            prev_x, prev_y = x, y
        
        return sorted(roots)
    
    def get_gauss_hermite_weights(self, roots=None):
        if roots is None:
            roots = self.get_roots()
        n = self.degree
        h_n_minus_1 = HermiteMPMath(n - 1, self.convention, dps=self._dps)
        weights = []
        for root in roots:
            h_at_root = h_n_minus_1.evaluate(root)
            numerator = mp.power(2, n - 1) * mp.factorial(n) * mp.sqrt(mp.pi)
            denominator = n * n * h_at_root * h_at_root
            weights.append(numerator / denominator)
        return weights
    """
    # BEGIN GPT Golub-Welsh
    def get_roots(self):
        """Compute roots using Golub–Welsch algorithm (spectral method)."""
        if self.degree == 0:
            return []
        
        n = self.degree
        
        # Build symmetric tridiagonal Jacobi matrix
        J = mp.matrix(n)
        # further initialization
        # J = mp.zeros(n)
        
        for i in range(n):
            for j in range(n):
                J[i, j] = mp.mpf(0)
        
        for i in range(n - 1):
            if self.convention == "physicist":
                val = mp.sqrt((i + 1) / 2)
            else:
                # probabilist Hermite
                val = mp.sqrt(i + 1)
            
            J[i, i + 1] = val
            J[i + 1, i] = val
        
        # Compute eigenvalues (roots)
        eigenvals = mp.eigsy(J, eigvals_only=True)
        
        # Ensure sorted output (eigsy usually returns sorted, but enforce it)
        roots = sorted(eigenvals)
        
        return roots
    
    def get_gauss_hermite_weights(self, roots=None):
        """Compute weights using Golub–Welsch eigenvectors."""
        if self.degree == 0:
            return []
        
        n = self.degree
        
        # Build Jacobi matrix
        J = mp.matrix(n)
        for i in range(n):
            for j in range(n):
                J[i, j] = mp.mpf(0)
        
        for i in range(n - 1):
            if self.convention == "physicist":
                val = mp.sqrt((i + 1) / 2)
            else:
                val = mp.sqrt(i + 1)
            
            J[i, i + 1] = val
            J[i + 1, i] = val
        
        # Eigen decomposition
        eigenvals, eigenvecs = mp.eigsy(J)
        
        # First row of eigenvectors → weights
        weights = []
        for i in range(n):
            v0 = eigenvecs[0, i]
            weights.append(mp.sqrt(mp.pi) * v0**2)
        
        # Sort (match eigenvalue ordering)
        paired = sorted(zip(eigenvals, weights), key=lambda x: x[0])
        weights = [w for _, w in paired]
        
        return weights
    # END GPT Golub-Welsh

    @classmethod
    def from_coefficients(cls, coefficients, convention: str = "physicist", dps: int = 50):
        return cls(degree=len(coefficients) - 1, convention=convention, coeffs=coefficients, dps=dps)
    
    @classmethod
    def from_symbolic(cls, symbolic_poly, dps: int = 50):
        coeff_strings = symbolic_poly.to_high_precision_coeffs(dps=dps)
        return cls.from_coefficients(coeff_strings, symbolic_poly.convention, dps)
    
    def to_numerical_coeffs(self):
        return [float(c) for c in self._coeffs]

def hermite_high_precision_basis(n_max: int, convention: str = "physicist", dps: int = 50):
    return [HermiteMPMath(n, convention, dps=dps) for n in range(n_max + 1)]