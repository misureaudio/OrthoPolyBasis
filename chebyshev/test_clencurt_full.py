import numpy as np
import math
from .clencurt import clencurt
from .quadrature_roots import ChebyshevQuadrature

def test_clencurt_comprehensive():
    print("=== Comprehensive Clenshaw-Curtis FFT Test ===")
    print()
    
    cq = ChebyshevQuadrature()
    
    # Test multiple n values
    for n in [4, 8, 16, 32, 64]:
        print(f"--- n={n} ---")
        
        # FFT method
        nodes_fft, weights_fft = clencurt(n)
        
        # O(n²) method
        nodes_on2 = np.array(cq.get_extrema_points(n))
        weights_on2 = np.array(cq._compute_clenshaw_curtis_weights(n))
        
        # Compare weight sums
        sum_fft = np.sum(weights_fft)
        sum_on2 = np.sum(weights_on2)
        print(f"  Weight sum: FFT={sum_fft:.15f}, O(n²)={sum_on2:.15f}")
        assert abs(sum_fft - 2.0) < 1e-14, f"FFT weight sum failed for n={n}"
        
        # Compare weights element-wise
        max_diff = np.max(np.abs(weights_fft - weights_on2))
        print(f"  Max weight difference: {max_diff:.2e}")
        assert max_diff < 1e-14, f"Weight mismatch for n={n}"
        
        # Test integration
        f = lambda x: np.exp(x)
        result_fft = np.sum(weights_fft * f(nodes_fft))
        result_on2 = np.sum(weights_on2 * f(nodes_on2))
        expected = math.e - 1/math.e
        
        print(f"  ∫ e^x dx: FFT={result_fft:.15f}, O(n²)={result_on2:.15f}")
        print(f"            Expected={expected:.15f}")
        
        # Both methods should agree exactly
        assert abs(result_fft - result_on2) < 1e-14, "Integration mismatch"
        print("  ✓ FFT and O(n²) methods agree")
        print()
    
    # Test accuracy with larger n
    print("--- Accuracy test (n=64) ---")
    nodes_fft, weights_fft = clencurt(64)
    result_fft = np.sum(weights_fft * np.exp(nodes_fft))
    expected = math.e - 1/math.e
    error = abs(result_fft - expected)
    print(f"  ∫ e^x dx: {result_fft:.15f}")
    print(f"  Expected: {expected:.15f}")
    print(f"  Error:    {error:.2e}")
    assert error < 1e-10, "Accuracy failed"
    print("  ✓ Accuracy test passed")
    print()
    
    # Performance comparison
    import time
    print("=== Performance Comparison ===")
    n = 1024
    
    start = time.perf_counter()
    for _ in range(10):
        nodes_fft, weights_fft = clencurt(n)
    fft_time = (time.perf_counter() - start) / 10 * 1000
    
    start = time.perf_counter()
    for _ in range(10):
        nodes_on2 = np.array(cq.get_extrema_points(n))
        weights_on2 = np.array(cq._compute_clenshaw_curtis_weights(n))
    on2_time = (time.perf_counter() - start) / 10 * 1000
    
    print(f"n={n} points:")
    print(f"  FFT method:   {fft_time:.3f} ms")
    print(f"  O(n²) method: {on2_time:.3f} ms")
    print(f"  Speedup:      {on2_time/fft_time:.1f}x")
    
    print()
    print("=" * 50)
    print("ALL COMPREHENSIVE TESTS PASSED!")

if __name__ == "__main__":
    test_clencurt_comprehensive()
