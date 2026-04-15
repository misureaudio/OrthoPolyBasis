import numpy as np
from scipy.special import eval_legendre

def naive_eval(series_coeffs, x):
    result = np.zeros_like(x)
    for k, c in enumerate(series_coeffs):
        result += c * eval_legendre(k, x)
    return result

print("Manual verification of 1*P_0 + 2*P_1 + 3*P_2 at x=0.5:")
x = 0.5
p0 = 1.0
p1 = x
p2 = (3*x*p1 - 1*p0)/2
print(f"  P_0({x}) = {p0}")
print(f"  P_1({x}) = {p1}")
print(f"  P_2({x}) = {p2}")
result = 1.0*p0 + 2.0*p1 + 3.0*p2
print(f"  1*P_0 + 2*P_1 + 3*P_2 = {result}")
