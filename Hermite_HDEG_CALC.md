# Task

What algorithms eventually exist to evalute very high degree Hermite polynomials, in case onewants a direct verification of how good  are the found roots?

## Algorithms for High-Degree Hermite Evaluation

### 1. Three-Term Recurrence (Forward) - Most Stable

The physicist's Hermite recurrence:
$$H_{n+1}(x) = 2x \cdot H_n(x) - 2n \cdot H_{n-1}(x)$$

Problem: Values grow as $H_n(0) \sim (-1)^{n/2} (n-1)!!$ for even n, causing overflow.

### 2. Miller's Algorithm (Backward Recurrence) - Best for Verification

Computes normalized ratios that stay bounded:
$$\frac{H_{n-k}(x)}{H_n(x)} \text{ via backward recurrence}$$

### 3. Using the Differential Equation

Since $H_n''(x) - 2x H_n'(x) + 2n H_n(x) = 0$, at a root:
$$\frac{H_n''(r)}{H_n'(r)} = 2r \quad \text{(if } r \text{ is truly a root)}$$

### 4. Logarithmic Evaluation

Compute $$\log|H_n(x)|$$ to avoid overflow, then exponentiate only when needed.