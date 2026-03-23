import numpy as np
import math


def M(t):
    """Функція вологості ґрунту"""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)


def dM_exact(t):
    """Точна аналітична похідна функції M(t)"""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


t0 = 1.0
exact_val = dM_exact(t0)
print("=" * 50)
print(f"1. Точне значення похідної в точці t0={t0}: {exact_val:.6f}")
print("=" * 50)


def diff_central(func, x, h):
    """Формула апроксимації похідної (центральна різниця)"""
    if h == 0:
        return float('inf')
    return (func(x + h) - func(x - h)) / (2 * h)


h_values = [10 ** i for i in range(-20, 4)]
best_h = None
min_error = float('inf')
best_approx = None

print("\n2. Дослідження залежності похибки від кроку h:")
for h_test in h_values:
    try:
        approx_val = diff_central(M, t0, h_test)
        error = abs(approx_val - exact_val)

        if error < min_error and error > 0 and not math.isnan(error):
            min_error = error
            best_h = h_test
            best_approx = approx_val
    except Exception:
        continue

print(f"Найкраща точність досягається при h0 = {best_h:.1e}")
print(f"Досягнута точність (похибка) R0 = {min_error:.6e}")
print("=" * 50)

h = 1e-3
print(f"\n3. Вибрано робочий крок h = {h}")

y_prime_h = diff_central(M, t0, h)
y_prime_2h = diff_central(M, t0, 2 * h)
print(f"4. Похідна з кроком h:  {y_prime_h:.10f}")
print(f"   Похідна з кроком 2h: {y_prime_2h:.10f}")

R1 = abs(y_prime_h - exact_val)
print(f"5. Похибка при кроці h (R1): {R1:.10e}")
print("=" * 50)

y_prime_R = y_prime_h + (y_prime_h - y_prime_2h) / 3
# Обчислення похибки
R2 = abs(y_prime_R - exact_val)

print(f"\n6. Метод Рунге-Ромберга:")
print(f"   Уточнене значення: {y_prime_R:.10f}")
print(f"   Похибка (R2): {R2:.10e}")
if R2 < R1:
    print("   Характер зміни похибки: Метод Рунге-Ромберга успішно зменшив похибку порівняно з базовою формулою.")
print("=" * 50)

y_prime_4h = diff_central(M, t0, 4 * h)

# Метод Ейткена (уточнене значення)
numerator = (y_prime_2h) ** 2 - y_prime_4h * y_prime_h
denominator = 2 * y_prime_2h - (y_prime_4h + y_prime_h)

if denominator != 0:
    y_prime_E = numerator / denominator
else:
    y_prime_E = float('nan')

ratio = abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h))
if ratio > 0:
    p = (1 / np.log(2)) * np.log(ratio)
else:
    p = float('nan')

R3 = abs(y_prime_E - exact_val)

print(f"\n7. Метод Ейткена:")
print(f"   Похідна з кроком 4h: {y_prime_4h:.10f}")
print(f"   Уточнене значення: {y_prime_E:.10f}")
print(f"   Порядок точності (p): {p:.4f}")
print(f"   Похибка (R3): {R3:.10e}")

if R3 < R1:
    print("   Характер зміни похибки: Метод Ейткена підвищив точність результату.")

print("\n--- Висновок щодо оптимальних режимів поливу ---")
print("Оскільки перша похідна описує швидкість зміни вологості (швидкість висихання ґрунту),")
print("систему автоматичного поливу доцільно налаштувати так, щоб вона вмикалася, коли:")
print("1. Сама вологість M(t) опускається нижче заданого критичного порогу.")
print("2. Швидкість висихання (від'ємне значення похідної M'(t)) стає занадто високою,")
print("   що свідчить про стрімку втрату вологи рослиною.")