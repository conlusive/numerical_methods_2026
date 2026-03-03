import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os


# 1. Зчитування даних
def read_data(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    x, y = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Dataset size']))
            y.append(float(row['Train time (sec)']))
    return np.array(x), np.array(y)


x_data, y_data = read_data('data.csv')


# 2. Метод Ньютона
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


def newton_poly(coef, x_data, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p


# --- 3. Метод Лагранжа
def lagrange_poly(x_nodes, y_nodes, x_val):
    result = 0.0
    n = len(x_nodes)
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


# Підготовка: Сплайн як "еталон" для досліджень похибок
true_func = CubicSpline(x_data, y_data)
# Точки для плавного малювання графіків
x_vals_plot = np.linspace(min(x_data), max(x_data), 500)

# Розрахунок коефіцієнтів Ньютона для основного графіку
coef = divided_differences(x_data, y_data)
x_target = 120000
y_pred = newton_poly(coef, x_data, x_target)

print(f"Прогноз для 120000: {y_pred:.4f} сек")

# ГРАФІК 1: ОСНОВНИЙ ПРОГНОЗ
plt.figure(figsize=(10, 6))
plt.title("Графік 1: Інтерполяція Ньютона та Прогноз")

# Малюємо лінію полінома
y_vals_newton = [newton_poly(coef, x_data, xi) for xi in x_vals_plot]
plt.plot(x_vals_plot, y_vals_newton, label="Поліном Ньютона", color='blue')

# Малюємо точки
plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label="Дані з CSV")
plt.scatter(x_target, y_pred, color='green', marker='X', s=150, zorder=6, label=f"Прогноз ({y_pred:.1f})")

plt.xlabel("Розмір датасету")
plt.ylabel("Час (сек)")
plt.legend()
plt.grid(True)

# ГРАФІК 2: ЕФЕКТ РУНГЕ (Фіксований інтервал)
plt.figure(figsize=(10, 6))
plt.title("Графік 2: Ефект Рунге (Фіксований інтервал [10k, 160k])")

for n in [5, 10, 20]:
    x_nodes = np.linspace(min(x_data), max(x_data), n)
    y_nodes = true_func(x_nodes)

    coef_n = divided_differences(x_nodes, y_nodes)
    y_interp = [newton_poly(coef_n, x_nodes, xi) for xi in x_vals_plot]

    error = np.abs(true_func(x_vals_plot) - y_interp)
    plt.plot(x_vals_plot, error, label=f"n={n} вузлів")

plt.xlabel("Розмір датасету")
plt.ylabel("Похибка (лог. шкала)")
plt.yscale("log")
plt.legend()
plt.grid(True)

# ГРАФІК 3: ЗМІННИЙ ІНТЕРВАЛ (Фіксований крок)
plt.figure(figsize=(10, 6))
plt.title("Графік 3: Фіксований крок h=10k (Змінний інтервал)")

h = 10000
start = min(x_data)

for n in [5, 10, 15]:
    # Формуємо вузли зі сталим кроком
    x_nodes = np.array([start + i * h for i in range(n)])
    y_nodes = true_func(x_nodes)

    coef_n = divided_differences(x_nodes, y_nodes)

    # Малюємо тільки в межах поточного інтервалу
    current_x_plot = np.linspace(min(x_nodes), max(x_nodes), 200)
    y_interp = [newton_poly(coef_n, x_nodes, xi) for xi in current_x_plot]

    error = np.abs(true_func(current_x_plot) - y_interp)
    plt.plot(current_x_plot, error, label=f"n={n} (до {int(x_nodes[-1])})")

plt.xlabel("Розмір датасету")
plt.ylabel("Похибка (лог. шкала)")
plt.yscale("log")
plt.legend()
plt.grid(True)

# ГРАФІК 4: ВІЗУАЛЬНЕ ПОРІВНЯННЯ (Дві лінії)
plt.figure(figsize=(10, 6))
plt.title("Графік 4: Візуальне порівняння Ньютона та Лагранжа")

# 1. Малюємо Ньютона (товста синя лінія)
# Ми вже рахували y_vals_newton для першого графіка, беремо їх звідти
plt.plot(x_vals_plot, y_vals_newton, label="Метод Ньютона", color='blue', linewidth=5, alpha=0.3)

# 2. Малюємо Лагранжа (червоний пунктир поверх)
y_vals_lagrange = [lagrange_poly(x_data, y_data, xi) for xi in x_vals_plot]
plt.plot(x_vals_plot, y_vals_lagrange, label="Метод Лагранжа", color='red', linestyle='--', linewidth=2)

plt.xlabel("Розмір датасету")
plt.ylabel("Час (сек)")
plt.legend()
plt.grid(True)

plt.show()
