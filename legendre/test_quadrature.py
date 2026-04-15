"""
Tests for Gauss-Legendre Quadrature
"""

import unittest
import numpy as np
from legendre.quadrature import (
    GaussLegendreQuadrature,
    gauss_legendre,
    gauss_legendre_newton,
    gauss_legendre_golub_welsch,
)


class TestGaussLegendreNewton(unittest.TestCase):
    """Test Newton-Raphson quadrature computation."""
    
    def test_n1(self):
        """Test n=1: node at 0, weight = 2."""
        nodes, weights = GaussLegendreQuadrature.newton_raphson(1)
        self.assertEqual(len(nodes), 1)
        self.assertAlmostEqual(float(nodes[0]), 0.0)
        self.assertAlmostEqual(float(weights[0]), 2.0)
    
    def test_n2(self):
        """Test n=2: nodes at ±1/√3."""
        nodes, weights = GaussLegendreQuadrature.newton_raphson(2)
        self.assertEqual(len(nodes), 2)
        expected_node = 1.0 / np.sqrt(3.0)
        self.assertAlmostEqual(abs(float(nodes[0])), expected_node, places=12)
        self.assertAlmostEqual(abs(float(nodes[1])), expected_node, places=12)
        self.assertAlmostEqual(float(weights[0]), 1.0)
        self.assertAlmostEqual(float(weights[1]), 1.0)
    
    def test_n3(self):
        """Test n=3: known analytical values."""
        nodes, weights = GaussLegendreQuadrature.newton_raphson(3)
        self.assertEqual(len(nodes), 3)
        # Known values for n=3
        expected_nodes = [-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)]
        expected_weights = [5.0/9.0, 8.0/9.0, 5.0/9.0]
        for i in range(3):
            self.assertAlmostEqual(float(nodes[i]), expected_nodes[i], places=12)
            self.assertAlmostEqual(float(weights[i]), expected_weights[i], places=12)
    
    def test_nodes_in_range(self):
        """Verify all nodes are in (-1, 1)."""
        for n in [5, 10, 20, 50]:
            nodes, _ = GaussLegendreQuadrature.newton_raphson(n)
            self.assertTrue(np.all(nodes > -1) and np.all(nodes < 1),
                          f"Nodes out of range for n={n}")
    
    def test_nodes_sorted(self):
        """Verify nodes are sorted in ascending order."""
        for n in [5, 10, 20]:
            nodes, _ = GaussLegendreQuadrature.newton_raphson(n)
            self.assertTrue(np.all(np.diff(nodes) > 0), f"Nodes not sorted for n={n}")
    
    def test_weights_positive(self):
        """Verify all weights are positive."""
        for n in [5, 10, 20, 50]:
            _, weights = GaussLegendreQuadrature.newton_raphson(n)
            self.assertTrue(np.all(weights > 0), f"Negative weights for n={n}")
    
    def test_weight_sum(self):
        """Verify sum of weights equals 2 (length of interval [-1,1])."""
        for n in [5, 10, 20, 50]:
            _, weights = GaussLegendreQuadrature.newton_raphson(n)
            weight_sum = np.sum(weights)
            self.assertAlmostEqual(weight_sum, 2.0, places=12,
                                msg=f"Weight sum != 2 for n={n}")


class TestGaussLegendreGolubWelsch(unittest.TestCase):
    """Test Golub-Welsch eigenvalue method."""
    
    def test_n3(self):
        """Test n=3 against known values."""
        nodes, weights = GaussLegendreQuadrature.golub_welsch(3)
        expected_nodes = [-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)]
        expected_weights = [5.0/9.0, 8.0/9.0, 5.0/9.0]
        for i in range(3):
            self.assertAlmostEqual(float(nodes[i]), expected_nodes[i], places=12)
            self.assertAlmostEqual(float(weights[i]), expected_weights[i], places=12)
    
    def test_agreement_with_newton(self):
        """Verify Golub-Welsch agrees with Newton-Raphson."""
        for n in [5, 10, 20]:
            nodes_nr, weights_nr = GaussLegendreQuadrature.newton_raphson(n)
            nodes_gw, weights_gw = GaussLegendreQuadrature.golub_welsch(n)
            
            max_node_diff = np.max(np.abs(nodes_nr - nodes_gw))
            max_weight_diff = np.max(np.abs(weights_nr - weights_gw))
            
            self.assertLess(max_node_diff, 1e-12, f"Node mismatch for n={n}")
            self.assertLess(max_weight_diff, 1e-12, f"Weight mismatch for n={n}")
    
    def test_high_n_stability(self):
        """Test that Golub-Welsch works for high n."""
        # This should work without numerical issues
        nodes, weights = GaussLegendreQuadrature.golub_welsch(100)
        self.assertEqual(len(nodes), 100)
        self.assertTrue(np.all(weights > 0))
        self.assertAlmostEqual(np.sum(weights), 2.0, places=10)


class TestQuadratureIntegration(unittest.TestCase):
    """Test numerical integration using quadrature."""
    
    def test_polynomial_exactness(self):
        """Gauss-Legendre with n points exactly integrates degree 2n-1 polynomials."""
        # Test: ?_{-1}^{1} x^k dx
        for n in [3, 5, 10]:
            nodes, weights = gauss_legendre(n)
            
            # Test all powers up to 2n-1
            for k in range(2 * n):
                # Exact integral of x^k from -1 to 1
                if k % 2 == 0:
                    exact = 2.0 / (k + 1)
                else:
                    exact = 0.0
                
                # Quadrature approximation
                approx = np.sum(weights * (nodes ** k))
                
                # Use absolute error for odd powers (exact=0), relative otherwise
                if exact == 0.0:
                    self.assertLess(abs(approx), 1e-12,
                                  f"Failed for x^{k} with n={n}: {approx} vs {exact}")
                else:
                    rel_error = abs(approx - exact) / abs(exact)
                    self.assertLess(rel_error, 1e-14,
                                  f"Failed for x^{k} with n={n}: {approx} vs {exact}")
    
    def test_integrate_function(self):
        """Test integrate method."""
        nodes, weights = gauss_legendre(10)
        
        # Integrate exp(x) from -1 to 1
        f = lambda x: np.exp(x)
        result = GaussLegendreQuadrature.integrate(f, nodes, weights)
        exact = np.e - 1/np.e  # e - 1/e
        
        self.assertAlmostEqual(result, exact, places=10)
    
    def test_integrate_transformed(self):
        """Test integration on arbitrary interval [a, b]."""
        nodes, weights = gauss_legendre(10)
        
        # Integrate x² from 0 to π
        f = lambda x: x * x
        a, b = 0.0, np.pi
        result = GaussLegendreQuadrature.integrate_transformed(f, a, b, nodes, weights)
        exact = (b**3 - a**3) / 3.0  # π³/3
        
        self.assertAlmostEqual(result, exact, places=9)
    
    def test_sine_integral(self):
        """Integrate sin(x) from 0 to π."""
        nodes, weights = gauss_legendre(20)
        f = lambda x: np.sin(x)
        result = GaussLegendreQuadrature.integrate_transformed(f, 0, np.pi, nodes, weights)
        exact = 2.0  # ?₀^π sin(x) dx = 2
        
        self.assertAlmostEqual(result, exact, places=12)
    
    def test_gaussian_integral(self):
        """Approximate Gaussian integral."""
        nodes, weights = gauss_legendre(50)
        f = lambda x: np.exp(-x * x)
        result = GaussLegendreQuadrature.integrate_transformed(f, -4, 4, nodes, weights)
        exact = np.sqrt(np.pi)  # Approximately 1.77245...
        
        self.assertAlmostEqual(result, exact, places=7)


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def test_gauss_legendre_default(self):
        """Default method is golub_welsch."""
        nodes1, weights1 = gauss_legendre(5)
        nodes2, weights2 = gauss_legendre_golub_welsch(5)
        self.assertTrue(np.allclose(nodes1, nodes2))
        self.assertTrue(np.allclose(weights1, weights2))
    
    def test_all_methods_same_result(self):
        """All methods produce same results."""
        nr_nodes, nr_weights = gauss_legendre_newton(5)
        gw_nodes, gw_weights = gauss_legendre_golub_welsch(5)
        default_nodes, default_weights = gauss_legendre(5)
        
        self.assertTrue(np.allclose(nr_nodes, gw_nodes))
        self.assertTrue(np.allclose(nr_weights, gw_weights))


if __name__ == "__main__":
    unittest.main()
