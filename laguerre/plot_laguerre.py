"""Plot Laguerre polynomials L_n(x) for n = 1 to 9."""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt


def evaluate_array(n: int, x_values, alpha: float = 0.0):
    """Vectorized evaluation of Laguerre polynomials using backward recurrence."""
    try:
        import numpy as np
        x_arr = np.asarray(x_values, dtype=np.float64)
        if n == 0: 
            return np.ones_like(x_arr)
        if n == 1: 
            return (alpha + 1) - x_arr
        u1, u0 = np.zeros_like(x_arr), np.ones_like(x_arr)
        for k in range(n-1, -1, -1):
            b = (2*k + alpha + 1 - x_arr)/(k+1)
            c = -(k + alpha)/(k+1)
            u1, u0 = u0.copy(), b*u0 + c*u1
        return u0
    except ImportError:
        raise ImportError('numpy required')


def main():
    # Typical interval for Laguerre polynomials: [0, ~25]
    # The roots of L_n(x) are roughly in [0, 4n], so this covers most behavior
    x = np.linspace(0, 25, 1000)

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 9))

    for n in range(1, 10):
        # Use vectorized evaluation via numpy
        y = evaluate_array(n, x, alpha=0.0)
        label = f'L_{{{n}}}(x)'
        plt.plot(x, y, color=colors[n-1], linewidth=2, label=label)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.xlabel('x')
    plt.ylabel(r'$L_n(x)$')
    plt.title('Laguerre Polynomials $L_n(x)$ for n = 1 to 9')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Set reasonable y-limits based on max values
    all_y = np.concatenate([evaluate_array(n, x, 0.0) for n in range(1, 10)])
    plt.ylim(-max(abs(all_y)) * 0.2, max(abs(all_y)) * 0.8)

    plt.tight_layout()
    plt.savefig('laguerre_polynomials.png', dpi=150)
    print("Plot saved to 'laguerre_polynomials.png'")


if __name__ == '__main__':
    main()