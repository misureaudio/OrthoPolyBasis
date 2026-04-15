import numpy as np

def clencurt(n):
    """
    Computes Clenshaw-Curtis nodes and weights using Trefethen's 
    O(n log n) method via FFT.
    
    Args:
        n: The degree (number of intervals). n should be even for 
           the standard symmetric implementation.
           
    Returns:
        nodes: x_j = cos(j*pi/n)
        weights: w_j such that sum(w * f(x)) approximates the integral.
    """
    if n == 0:
        return np.array([1.0]), np.array([2.0])
    if n == 1:
        return np.array([1.0, -1.0]), np.array([1.0, 1.0])
    
    # Clenshaw-Curtis nodes
    theta = np.pi * np.arange(n + 1) / n
    nodes = np.cos(theta)
    
    # 1. Compute the integrals of the Chebyshev basis functions T_k(x)
    # The integral of T_k(x) from -1 to 1 is 2/(1-k^2) for even k, 0 for odd.
    c = np.zeros(n + 1)
    c[0::2] = 2.0 / (1.0 - np.arange(0, n + 1, 2)**2)
    
    # 2. Create the periodic extension for the IFFT (length 2n)
    # The vector must be: [c0, c1, ..., cn, cn-1, ..., c1]
    # This corresponds to the coefficients for a Type-I Discrete Cosine Transform.
    v = np.concatenate([c, c[n-1:0:-1]])
    
    # 3. Compute weights using IFFT
    # NumPy's IFFT computes: 1/N * sum(v_k * exp(i*2pi*jk/N))
    w_full = np.real(np.fft.ifft(v))
    
    # 4. Extract weights
    # Because of the symmetry in the periodic extension:
    # - The endpoints (j=0, j=n) appear once in the period.
    # - The interior points (j=1...n-1) appear twice.
    # To get the quadrature weights, we take the IFFT result and 
    # double the interior values.
    weights = np.zeros(n + 1)
    weights[0] = w_full[0]
    weights[n] = w_full[n]
    weights[1:n] = 2.0 * w_full[1:n]
    
    return nodes, weights

def clencurt_quadrature(f, n: int):
    """
    Perform numerical integration using Clenshaw-Curtis quadrature.

    Args:
        f: Function to integrate over [-1, 1]
        n: The degree (number of intervals). Should be even for standard symmetric implementation.

    Returns:
        Approximation of ∫_{-1}^{1} f(x) dx
    """
    nodes, weights = clencurt(n)
    return np.sum(weights * f(nodes))

# === Verification ===

def run_test(n):
    x, w = clencurt(n)
    print(f"--- Testing n={n} ---")
    print(f"Nodes: {x}")
    print(f"Weights: {w}")
    
    sum_w = np.sum(w)
    print(f"Sum of weights: {sum_w:.14f} (Expected: 2.0)")
    
    # Test ∫ x² dx from -1 to 1 (Expected 2/3 ≈ 0.66666666666667)
    int_x2 = np.sum(w * x**2)
    print(f"∫x² dx: {int_x2:.14f}")
    
    # Test ∫ x⁴ dx from -1 to 1 (Expected 2/5 = 0.4)
    int_x4 = np.sum(w * x**4)
    print(f"∫x⁴ dx: {int_x4:.14f}")
    
    assert abs(sum_w - 2.0) < 1e-14
    assert abs(int_x2 - 2/3) < 1e-14

# run_test(4)