import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 1. Визначення функції
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


# 2. Знаходження точного значення інтегралу
a, b = 0, 24
I_0, _ = quad(f, a, b)
print(f"Точне значення інтегралу I_0: {I_0:.15f}")


# 3. Функція для методу Сімпсона
def simpson_method(f, a, b, N):
    if N % 2 != 0: N += 1  # N має бути парним
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)

    # Формула: (h/3) * (f0 + 4*sum(odd) + 2*sum(even) + fN)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S


# 4. Дослідження залежності точності від N
Ns = np.arange(10, 1001, 10)
errors = [abs(simpson_method(f, a, b, n) - I_0) for n in Ns]

# Пошук N_opt для точності 1e-12
target_eps = 1e-12
N_opt = 10
while abs(simpson_method(f, a, b, N_opt) - I_0) > target_eps and N_opt < 10000:
    N_opt += 2
eps_opt = abs(simpson_method(f, a, b, N_opt) - I_0)

print(f"Оптимальне N (для 1e-12): {N_opt}, досягнута точність: {eps_opt:.2e}")

# 5. Похибка при N0 = N_opt / 10 (кратне 8)
N0 = int(N_opt / 10)
N0 = (N0 // 8) * 8 if N0 >= 8 else 8
eps0 = abs(simpson_method(f, a, b, N0) - I_0)
print(f"N0: {N0}, Похибка eps0: {eps0:.2e}")

# 6. Метод Рунге-Ромберга (уточнення для N0 та N0/2)
I_h = simpson_method(f, a, b, N0)
I_2h = simpson_method(f, a, b, N0 // 2)
# Для методу Сімпсона (p=4) коефіцієнт (2^4 - 1) = 15
I_R = I_h + (I_h - I_2h) / 15
epsR = abs(I_R - I_0)
print(f"Похибка після Рунге-Ромберга: {epsR:.2e}")

# 7. Метод Ейткена
I1 = simpson_method(f, a, b, N0)
I2 = simpson_method(f, a, b, N0 // 2)
I3 = simpson_method(f, a, b, N0 // 4)
# Уточнене значення
I_Aitken = (I2 ** 2 - I1 * I3) / (2 * I2 - (I1 + I3))
epsA = abs(I_Aitken - I_0)
# Порядок точності p
p_aitken = np.log(abs((I3 - I2) / (2 * I2 - I1 - I2))) / np.log(2)  # спрощено для q=2
print(f"Похибка методу Ейткена: {epsA:.2e}")


# 9. Адаптивний алгоритм (рекурсивний)
def adaptive_simpson(f, a, b, eps, whole):
    mid = (a + b) / 2
    left = simpson_method(f, a, mid, 2)
    right = simpson_method(f, mid, b, 2)
    if abs(left + right - whole) <= 15 * eps:  # 15 - коефіцієнт для правила Рунге
        return left + right + (left + right - whole) / 15
    return adaptive_simpson(f, a, mid, eps / 2, left) + \
        adaptive_simpson(f, mid, b, eps / 2, right)


I_adaptive = adaptive_simpson(f, a, b, 1e-7, simpson_method(f, a, b, 2))
print(f"Результат адаптивного алгоритму (eps=1e-7): {I_adaptive:.10f}")

# Побудова графіка залежності похибки від N
plt.figure(figsize=(10, 5))
plt.semilogy(Ns, errors, 'r-', label='Simpson Error')
plt.axhline(y=target_eps, color='g', linestyle='--', label='Target 1e-12')
plt.title('Залежність похибки від числа розбиттів N')
plt.xlabel('N')
plt.ylabel('log(Error)')
plt.grid(True)
plt.legend()
plt.show()