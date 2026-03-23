import os
import csv
import matplotlib.pyplot as plt


# лінійна інтерполяція для знаходження y між вузлами
def get_y_true(x_val, x_nodes, y_nodes):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            return y_nodes[i] + (y_nodes[i + 1] - y_nodes[i]) * (x_val - x_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
    return y_nodes[-1]


# зчитування даних з csv
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # пропускаємо заголовок
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y


# формування матриці a
def form_matrix(x, m):
    a = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            a[i][j] = sum(xi ** (i + j) for xi in x)
    return a


# формування вектора вільних членів b
def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k] ** i) for k in range(len(x)))
    return b


# метод гауса з вибором головного елемента
def gauss_solve(a, b):
    n = len(a)
    a_copy = [row[:] for row in a]
    b_copy = b[:]

    # прямий хід
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(a_copy[i][k]) > abs(a_copy[max_row][k]):
                max_row = i

        # перестановка рядків
        a_copy[k], a_copy[max_row] = a_copy[max_row], a_copy[k]
        b_copy[k], b_copy[max_row] = b_copy[max_row], b_copy[k]

        for i in range(k + 1, n):
            if a_copy[k][k] == 0: continue
            factor = a_copy[i][k] / a_copy[k][k]
            for j in range(k, n):
                a_copy[i][j] -= factor * a_copy[k][j]
            b_copy[i] -= factor * b_copy[k]

    # зворотній хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(a_copy[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b_copy[i] - s) / a_copy[i][i]
    return x_sol


# обчислення значень полінома
def polynomial(x_vals, coef):
    return [sum(coef[i] * (xv ** i) for i in range(len(coef))) for xv in x_vals]


# обчислення дисперсії
def calculate_variance(y_true, y_approx):
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n


# головний блок програми
# визначаємо абсолютний шлях до папки зі скриптом
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data.csv')

x, y = read_data(data_path)

variances = []
max_degree = 10
n_nodes = len(x)

# вибір оптимального ступеня полінома
for m in range(1, max_degree + 1):
    a_mat = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(a_mat, b_vec)
    y_approx = polynomial(x, coef)
    var = calculate_variance(y, y_approx)
    variances.append(var)

# оптимальний степінь
optimal_m = variances.index(min(variances)) + 1

# побудова апроксимації для оптимального полінома
a_opt = form_matrix(x, optimal_m)
b_opt = form_vector(x, y, optimal_m)
coef_opt = gauss_solve(a_opt, b_opt)

# генерація точок для плавного графіка апроксимації
x_smooth = [x[0] + i * (x[-1] - x[0]) / 200 for i in range(201)]
y_smooth = polynomial(x_smooth, coef_opt)

# прогноз на наступні 3 місяці
x_future = [25, 26, 27]
y_future = polynomial(x_future, coef_opt)

# табулювання похибки з дрібним кроком h1
h1 = (x[-1] - x[0]) / (20 * n_nodes)
x_err = []
curr_x = x[0]
while curr_x <= x[-1]:
    x_err.append(curr_x)
    curr_x += h1

# побудова графіків
# графік 1: дисперсія від степеня
plt.figure(1, figsize=(8, 6))
plt.plot(range(1, max_degree + 1), variances, 'b-o')
plt.axvline(x=optimal_m, color='r', linestyle='--', label=f'оптимальне m={optimal_m}')
plt.title("залежність дисперсії від степеня")
plt.xlabel("степінь полінома (m)")
plt.ylabel("дисперсія")
plt.legend()
plt.grid(True)

# графік 2: фактичні дані, апроксимація та прогноз
plt.figure(2, figsize=(8, 6))
plt.plot(x, y, 'ko', label='фактичні дані')
plt.plot(x_smooth, y_smooth, 'b-', label=f'апроксимація (m={optimal_m})')
plt.plot(x_future, y_future, 'rx--', label='прогноз')
plt.title("апроксимація та прогноз")
plt.xlabel("місяць")
plt.ylabel("температура")
plt.legend()
plt.grid(True)

# графік 3: похибки для різних m на відрізку з кроком h1
plt.figure(3, figsize=(8, 6))
for m in range(1, max_degree + 1):
    a_mat = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    c_m = gauss_solve(a_mat, b_vec)

    y_approx_err = polynomial(x_err, c_m)
    y_true_err = [get_y_true(xv, x, y) for xv in x_err]
    error_vals = [abs(y_true_err[i] - y_approx_err[i]) for i in range(len(x_err))]

    if m == optimal_m:
        plt.plot(x_err, error_vals, 'r-', linewidth=2, label=f'm={m} (опт.)')
    else:
        plt.plot(x_err, error_vals, alpha=0.3)

plt.title("похибка апроксимації")
plt.xlabel("місяць")
plt.ylabel("абсолютна похибка")
plt.legend()
plt.grid(True)

# показати всі три вікна одночасно
plt.show()