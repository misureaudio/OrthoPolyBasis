"""Demonstration of the Hermite Polynomial Flexible Stack.

This module shows how to use all three layers together:
    - symbolic.py: Exact coefficients, algebraic manipulation
    - high_precision.py: Root finding, weights for quadrature  
    - numerical.py: Fast evaluation at many points
"""

import numpy as np
from hermite import (
    HermiteSymbolic,
    HermiteMPMath,
    HermitePolynomial,
    GaussHermiteQuadrature,
    HermiteProjection,
)

import math

def demo_symbolic_layer():
    """Demonstrate symbolic.py - The Source."""
    print("=" * 60)
    print("SYMBOLIC LAYER (The Source)")
    print("=" * 60)
    
    # Create H_5(x) with exact rational coefficients
    h5 = HermiteSymbolic(degree=5, convention="physicist")
    print(f"H_5(x) = {h5}")
    
    # Get exact coefficients as SymPy Rationals
    coeffs_exact = h5.get_coefficients()
    print(f"Exact coefficients: {coeffs_exact}")
    
    # Algebraic operations preserve exactness
    h5_squared = h5 * h5
    print(f"H_5(x)^2 degree: {h5_squared.degree}")
    
    # Derivative using identity d/dx H_n = 2n*H_{n-1}
    deriv = h5.derivative()
    print(f"d/dx H_5(x) = {deriv}")
    
    return h5


def demo_high_precision_layer(h5_symbolic):
    """Demonstrate high_precision.py - The Bridge."""
    print("\n" + "=" * 60)
    print("HIGH-PRECISION LAYER (The Bridge)")
    print("=" * 60)
    
    # Convert symbolic to high-precision
    h5_hp = HermiteMPMath.from_symbolic(h5_symbolic, dps=80)
    print(f"H_5 with 80 decimal places precision")
    
    # Find roots to extreme precision
    roots = h5_hp.get_roots()
    print(f"Roots of H_5(x): {roots}")
    
    # Compute quadrature weights
    weights = h5_hp.get_gauss_hermite_weights(roots)
    print(f"Quadrature weights: {weights}")
    
    # Evaluate at arbitrary point with high precision
    x_test = 1.23456789012345
    value = h5_hp.evaluate(x_test)
    print(f"H_5({x_test}) = {value}")
    
    return h5_hp, roots, weights


def demo_numerical_layer(h5_symbolic):
    """Demonstrate numerical.py - The Engine."""
    print("\n" + "=" * 60)
    print("NUMERICAL LAYER (The Engine)")
    print("=" * 60)
    
    # Convert symbolic to numerical for speed
    h5_num = HermitePolynomial.from_symbolic(h5_symbolic)
    print(f"H_5 as NumPy polynomial")
    
    # Fast evaluation at many points (broadcasting!)
    x_values = np.linspace(-4, 4, 1000)
    y_values = h5_num.evaluate(x_values)
    print(f"Evaluated H_5 at {len(x_values)} points in one call")
    
    # Verify against SciPy
    test_pts, our_vals, max_err = h5_num.verify_against_scipy()
    print(f"Max relative error vs SciPy: {max_err:.2e}")
    
    return h5_num


def demo_integration_layer():
    """Demonstrate integration.py - The Client."""
    print("\n" + "=" * 60)
    print("INTEGRATION LAYER (The Client)")
    print("=" * 60)
    
    # Create quadrature with high-precision roots/weights
    quad = GaussHermiteQuadrature(n=20, dps=80)
    print(f"Gauss-Hermite quadrature with {quad.n} points")
    
    # Integrate exp(-x^2/2) * exp(-x^2) = exp(-3x^2/2)
    # Exact: sqrt(2*pi/3)
    result = quad.integrate(lambda x: np.exp(-x**2 / 2))
    exact = np.sqrt(2 * np.pi / 3)
    print(f"Integral of exp(-x^2/2) with weight exp(-x^2):")
    print(f"  Computed: {result:.15f}")
    print(f"  Exact:    {exact:.15f}")
    print(f"  Error:    {abs(result - exact):.2e}")
    
    # Projection example
    projector = HermiteProjection(n_max=10, quadrature_points=50)
    f = lambda x: np.exp(-x**2 / 4)  # Function to expand
    coeffs = projector.project(f)
    print(f"\nHermite expansion coefficients for exp(-x^2/4):")
    print(f"First 6 coeffs: {coeffs[:6]}")
    
    return quad, projector


def demo_flexible_stack():
    """Demonstrate the complete flexible stack workflow."""
    print("\n" + "=" * 60)
    print("FLEXIBLE STACK - Complete Workflow")
    print("=" * 60)
    
    # Step 1: Symbolic - Generate exact polynomial
    print("\n[1] SYMBOLIC: Generate H_10 with exact coefficients")
    h10_sym = HermiteSymbolic(10, "physicist")
    leading_coeff = h10_sym.get_leading_coefficient()
    print(f"    Leading coefficient (exact): {leading_coeff} = 2^10")
    
    # Step 2: High-precision - Find roots for quadrature
    print("\n[2] HIGH-PRECISION: Compute roots to 60 decimal places")
    h10_hp = HermiteMPMath.from_symbolic(h10_sym, dps=60)
    roots = h10_hp.get_roots()
    print(f"    Found {len(roots)} roots")
    print(f"    First root: {roots[0]}")
    
    # Step 3: Numerical - Fast evaluation for plotting
    print("\n[3] NUMERICAL: Evaluate at 10,000 points")
    h10_num = HermitePolynomial.from_symbolic(h10_sym)
    x_plot = np.linspace(-5, 5, 10000)
    y_plot = h10_num.evaluate(x_plot)
    print(f"    Range: [{y_plot.min():.2f}, {y_plot.max():.2f}]")
    
    # Step 4: Integration - Use roots for quadrature
    print("\n[4] INTEGRATION: Verify orthogonality")
    quad = GaussHermiteQuadrature(10, dps=60)
    h10_vals = h10_num.evaluate(quad.roots)
    norm_sq = np.sum(quad.weights * h10_vals ** 2)
    expected_norm_sq = 2**10 * math.factorial(10) * np.sqrt(np.pi)
    print(f"    ||H_10||^2 computed: {norm_sq:.6f}")
    print(f"    ||H_10||^2 exact:   {expected_norm_sq:.6f}")
    print(f"    Relative error:     {abs(norm_sq - expected_norm_sq)/expected_norm_sq:.2e}")


def demo_high_degree_stability(n):
    """Demonstrate handling of high-degree polynomials."""
    print("\n" + "=" * 60)
    print(f"HIGH-DEGREE STABILITY (n={n})")
    print("=" * 60)
    
    # For n > 100, float64 may overflow - use high precision
    print(f"\n[SYMBOLIC] Generate H_{n} exact coefficients")
    hn_sym = HermiteSymbolic(n, "physicist")
    # h150_sym = HermiteSymbolic(25, "physicist")
    leading_coeff = hn_sym.get_leading_coefficient()
    print(f"    Leading coefficient: 2^{n} = {leading_coeff}")
    
    # Convert to high precision (float64 would overflow!)
    print(f"\n[HIGH-PRECISION] Evaluate H_{n}(2.0) with dps=80")
    hn_hp = HermiteMPMath.from_symbolic(hn_sym, dps=80)
    value_at_2 = hn_hp.evaluate(2.0)
    print(f"    H_{n}(2.0) = {value_at_2}")
    
    # Numerical layer would overflow for this degree
    print(f"\n[NUMERICAL] Warning: float64 overflows for n={n}")
    print("    Use high_precision.py for degrees > 100")


if __name__ == "__main__":
    # Run all demonstrations
    h5 = demo_symbolic_layer()
    demo_high_precision_layer(h5)
    demo_numerical_layer(h5)
    demo_integration_layer()
    demo_flexible_stack()
    demo_high_degree_stability(24)
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
