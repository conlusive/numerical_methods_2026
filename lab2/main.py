import csv
import numpy as np
import matplotlib.pyplot as plt
import math


# ==========================================
# 1. ЗЧИТУВАННЯ ДАНИХ
# ==========================================
def read_data(filename):
    """Зчитування даних з CSV файлу згідно з методичкою."""
    x = []
    y = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            t_val = row.get('t') or row.get('Train time (sec)')  # Підтримка різних назв колонок
            y.append(float(t_val))
    return np.array(x), np.array(y)


# ==========================================
# 2. ІНТЕРПОЛЯЦІЙНИЙ МНОГОЧЛЕН НЬЮТОНА
# ==========================================
def divided_differences(x, y):
    """Обчислення таблиці розділених різниць."""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


def newton_polynomial(x_data, y_data, x):
    """Обчислення значення полінома Ньютона для заданого x."""
    coef = divided_differences(x_data, y_data)
    n = len(x_data)
    result = coef[0]

    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        result += term
    return result


# ==========================================
# 3. ФАКТОРІАЛЬНІ МНОГОЧЛЕНИ (Скінченні різниці)
# ==========================================
def finite_differences(y):
    """Обчислення таблиці скінченних різниць для рівновіддалених вузлів."""
    n = len(y)
    diffs = np.zeros([n, n])
    diffs[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = diffs[i + 1][j - 1] - diffs[i][j - 1]
    return diffs[0, :]


def factorial_polynomial(y_data, t):
    """Обчислення полінома через факторіальні многочлени."""
    diffs = finite_differences(y_data)
    n = len(y_data)
    result = diffs[0]

    for i in range(1, n):
        term = diffs[i] / math.factorial(i)
        for j in range(i):
            term *= (t - j)
        result += term
    return result


# ==========================================
# 4. ГОЛОВНА ФУНКЦІЯ ТА ДОСЛІДЖЕННЯ
# ==========================================
def main():
    # Зчитування даних
    x_data, y_data = read_data("data.csv")
    print(f"Вхідні дані (Розмір датасету): {x_data}")
    print(f"Вхідні дані (Час тренування): {y_data}")

    # Завдання 2: Оцінка часу для 120000
    x_target = 120000

    # Метод Ньютона
    pred_newton = newton_polynomial(x_data, y_data, x_target)
    print(f"\nПрогноз для {x_target} (Метод Ньютона): {pred_newton:.2f} сек")

    # Метод факторіальних многочленів
    # Оскільки факторіальні многочлени вимагають рівновіддалених вузлів,
    # ми генеруємо рівномірну сітку, знаходимо на ній значення (за допомогою Ньютона),
    # а потім застосовуємо формулу факторіального многочлена.
    h = (x_data[-1] - x_data[0]) / (len(x_data) - 1)
    x_uniform = np.linspace(x_data[0], x_data[-1], len(x_data))
    y_uniform = [newton_polynomial(x_data, y_data, xi) for xi in x_uniform]

    t_target = (x_target - x_data[0]) / h
    pred_factorial = factorial_polynomial(y_uniform, t_target)
    print(f"Прогноз для {x_target} (Факторіальні многочлени): {pred_factorial:.2f} сек")

    # ==========================================
    # ПОБУДОВА ГРАФІКІВ (Основна крива)
    # ==========================================
    x_plot = np.linspace(min(x_data), max(x_data), 500)
    y_newton_plot = [newton_polynomial(x_data, y_data, xi) for xi in x_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_newton_plot, label="Інтерполяція Ньютона", color='blue')
    plt.scatter(x_data, y_data, color='red', s=50, label="Експериментальні дані", zorder=5)
    plt.scatter([x_target], [pred_newton], color='green', s=80, marker='*', label=f"Прогноз ({x_target})", zorder=5)

    plt.title("Прогноз часу тренування моделі машинного навчання (Варіант 3)")
    plt.xlabel("Розмір датасету (n)")
    plt.ylabel("Час тренування, сек")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ==========================================
    # ДОСЛІДЖЕННЯ ЕФЕКТУ РУНГЕ (n=5, 10, 20)
    # ==========================================
    print("\n--- Дослідження ефекту Рунге та впливу кількості вузлів ---")

    # Для демонстрації ефекту Рунге використаємо базову функцію, яка описує наші дані
    # Апроксимуємо ідеальну гладку криву (f(x) ~ a * x^b)
    def true_function(x):
        return 8 * (x / 10000) ** 1.42

    a, b = min(x_data), max(x_data)
    nodes_list = [5, 10, 20]

    plt.figure(figsize=(12, 8))
    x_dense = np.linspace(a, b, 1000)
    plt.plot(x_dense, true_function(x_dense), 'k--', label="Гладкий тренд (еталон)", linewidth=2)

    for n in nodes_list:
        # Генеруємо вузли
        x_nodes = np.linspace(a, b, n)
        y_nodes = true_function(x_nodes)

        # Обчислюємо інтерполяцію Ньютона для цих вузлів
        y_interp = [newton_polynomial(x_nodes, y_nodes, xi) for xi in x_dense]

        # Табуляція з кроком h = (b-a)/20n
        step_h = (b - a) / (20 * n)
        x_tab = np.arange(a, b, step_h)
        y_tab = [newton_polynomial(x_nodes, y_nodes, xi) for xi in x_tab]

        # Графік
        plt.plot(x_dense, y_interp, label=f"Інтерполяція (n={n})")

        # Обчислення максимальної похибки
        error = [abs(true_function(xi) - newton_polynomial(x_nodes, y_nodes, xi)) for xi in x_tab]
        print(f"Максимальна похибка для n={n}: {max(error):.4f}")

    plt.title("Дослідження ефекту Рунге (Осциляції полінома при великій кількості вузлів)")
    plt.xlabel("Розмір датасету")
    plt.ylabel("Час тренування")
    plt.legend()
    plt.grid(True)
    plt.ylim(-100, 600)  # Обмежуємо вісь Y, щоб побачити сильні осциляції при n=20
    plt.show()

    print("\nЗапис результатів табуляції у файл results.txt...")
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("Розмір (n) | Прогноз часу (t)\n")
        f.write("-" * 35 + "\n")
        for xi, yi in zip(x_plot[::10], y_newton_plot[::10]):
            f.write(f"{xi:10.0f} | {yi:10.2f} сек\n")
    print("Готово!")



if __name__ == "__main__":
    main()