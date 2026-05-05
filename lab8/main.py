import numpy as np
import matplotlib.pyplot as plt


# =================================================================
# ЧАСТИНА 1: ТРАНСЦЕНДЕНТНІ РІВНЯННЯ
# =================================================================

def f(x):
    return 0.5 * x ** 2 - np.cos(x) - 2


def df(x):
    return x + np.sin(x)


def ddf(x):
    return 1 + np.cos(x)


def tabulate_and_save(a, b, step, filename="tabulation.txt"):
    """Пункт 1: Табуляція та пошук початкових наближень"""
    x_vals = np.arange(a, b + step, step)
    y_vals = f(x_vals)

    with open(filename, "w") as file:
        file.write("x\t\tf(x)\n")
        for x, y in zip(x_vals, y_vals):
            file.write(f"{x:.4f}\t{y:.4f}\n")

    approx_roots = []
    for i in range(len(y_vals) - 1):
        if y_vals[i] * y_vals[i + 1] < 0:
            approx_roots.append((x_vals[i] + x_vals[i + 1]) / 2)
    return approx_roots


def check_stop(x_next, x_curr, eps):
    """Пункт 3: Одночасне виконання двох умов зупинки"""
    return abs(f(x_next)) < eps and abs(x_next - x_curr) < eps


# Методи уточнення (Пункт 2)
def simple_iteration(x0, eps):
    tau = -1 / df(x0)
    x_curr, n = x0, 0
    while n < 500:
        x_next = x_curr + tau * f(x_curr)
        n += 1
        if check_stop(x_next, x_curr, eps): return x_next, n
        x_curr = x_next
    return x_curr, n


def newton_method(x0, eps):
    x_curr, n = x0, 0
    while n < 500:
        x_next = x_curr - f(x_curr) / df(x_curr)
        n += 1
        if check_stop(x_next, x_curr, eps): return x_next, n
        x_curr = x_next
    return x_curr, n


def chebyshev_method(x0, eps):
    x_curr, n = x0, 0
    while n < 500:
        fx, dfx, ddfx = f(x_curr), df(x_curr), ddf(x_curr)
        x_next = x_curr - fx / dfx - 0.5 * (fx ** 2 * ddfx) / (dfx ** 3)
        n += 1
        if check_stop(x_next, x_curr, eps): return x_next, n
        x_curr = x_next
    return x_curr, n


def secant_method(x0, x1, eps):
    n = 0
    while n < 500:
        x_next = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        n += 1
        if check_stop(x_next, x1, eps): return x_next, n
        x0, x1 = x1, x_next
    return x1, n


# =================================================================
# ЧАСТИНА 2: АЛГЕБРАЇЧНІ РІВНЯННЯ
# =================================================================

def save_coeffs(coeffs, filename="coeffs.txt"):
    with open(filename, "w") as f:
        f.write(" ".join(map(str, coeffs)))


def read_coeffs(filename="coeffs.txt"):
    with open(filename, "r") as f:
        return [float(c) for c in f.read().split()]


def horner_newton(coeffs, x0, eps):
    """Пункт 8: Метод Ньютона + Схема Горнера"""
    x_curr, n = x0, 0
    m = len(coeffs) - 1
    while n < 500:
        b = [0] * (m + 1)
        b[0] = coeffs[0]
        for i in range(1, m + 1): b[i] = coeffs[i] + x_curr * b[i - 1]

        c = [0] * m
        c[0] = b[0]
        for i in range(1, m): c[i] = b[i] + x_curr * c[i - 1]

        x_next = x_curr - b[m] / c[m - 1]
        n += 1
        if abs(x_next - x_curr) < eps: return x_next, n
        x_curr = x_next
    return x_curr, n


def lin_method(coeffs, p0, q0, eps):
    """Пункт 9: Метод Ліна для комплексних коренів"""
    p, q, n = p0, q0, 0
    m = len(coeffs) - 1
    while n < 1000:
        b = [0] * (m + 1)
        b[0] = coeffs[0]
        b[1] = coeffs[1] - p * b[0]
        for i in range(2, m + 1): b[i] = coeffs[i] - p * b[i - 1] - q * b[i - 2]

        p_next = b[m - 1] / b[m - 2]
        q_next = b[m] / b[m - 2]
        n += 1
        if abs(p_next - p) < eps and abs(q_next - q) < eps:
            disc = p_next ** 2 - 4 * q_next
            r1 = complex(-p_next / 2, np.sqrt(abs(disc)) / 2 if disc < 0 else 0)
            r2 = complex(-p_next / 2, -np.sqrt(abs(disc)) / 2 if disc < 0 else 0)
            return (r1, r2), n
        p, q = p_next, q_next
    return None, n


# =================================================================
# ГОЛОВНИЙ ЦИКЛ (ВИКОНАННЯ ВСІХ ПУНКТІВ)
# =================================================================

if __name__ == "__main__":
    EPS = 1e-4

    # 1. ТРАНСЦЕНДЕНТНА ЧАСТИНА
    print("--- 1. Табуляція та графік ---")
    approx_roots = tabulate_and_save(-5, 5, 0.5)

    # Побудова графіка
    x_p = np.linspace(-5, 5, 500)
    plt.figure(figsize=(10, 4))
    plt.plot(x_p, f(x_p), label='f(x)')
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Трансцендентна функція")
    plt.grid(True)
    plt.show()

    # Вибір двох точок: зростання та спадання
    r_up = [r for r in approx_roots if df(r) > 0][0]
    r_down = [r for r in approx_roots if df(r) < 0][0]

    print(f"\nКорені (Пункт 4):")
    print(f"Ньютон (зростання): {newton_method(r_up, EPS)}")
    print(f"Чебишев (спадання): {chebyshev_method(r_down, EPS)}")
    print(f"Хорди: {secant_method(r_up - 0.5, r_up, EPS)}")

    # 2. АЛГЕБРАЇЧНА ЧАСТИНА (Пункт 5)
    # x^3 - 1.5x^2 + 1.5x - 1 = 0 (Дійсний корінь 1.0)
    coeffs_alg = [1.0, -1.5, 1.5, -1.0]
    save_coeffs(coeffs_alg)

    print("\n--- 2. Алгебраїчне рівняння ---")
    x_alg = np.linspace(-1, 2.5, 500)
    plt.figure(figsize=(10, 4))
    plt.plot(x_alg, [np.polyval(coeffs_alg, i) for i in x_alg])
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Алгебраїчне рівняння (1 дійсний корінь)")
    plt.grid(True)
    plt.show()

    loaded = read_coeffs()
    print(f"Дійсний корінь (Горнер): {horner_newton(loaded, 1.5, EPS)}")
    print(f"Комплексні корені (Лін): {lin_method(loaded, 0.5, 0.5, EPS)}")
