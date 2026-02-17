import requests
import numpy as np
import matplotlib.pyplot as plt


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def solve_progonka(h, y):
    n = len(y) - 1
    alpha = np.zeros(n + 1)
    beta = np.zeros(n + 1)
    F = np.zeros(n + 1)
    for i in range(1, n):
        F[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    alpha[2] = -h[1] / (2 * (h[0] + h[1]))
    beta[2] = F[1] / (2 * (h[0] + h[1]))
    for i in range(2, n):
        denom = 2 * (h[i - 1] + h[i]) + h[i - 1] * alpha[i]
        alpha[i + 1] = -h[i] / denom
        beta[i + 1] = (F[i] - h[i - 1] * beta[i]) / denom
    c = np.zeros(n + 1)
    for i in range(n - 1, 0, -1):
        c[i] = alpha[i + 1] * c[i + 1] + beta[i + 1]
    return c


def get_spline_params(x, y):
    n = len(x) - 1
    h = np.diff(x)
    c_coeffs = solve_progonka(h, y)
    a = y[:-1]
    b = np.zeros(n);
    d = np.zeros(n)
    for i in range(n):
        d[i] = (c_coeffs[i + 1] - c_coeffs[i]) / (3 * h[i])
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (c_coeffs[i + 1] + 2 * c_coeffs[i])
    return a, b, c_coeffs[:-1], d


def interpolate(x_val, x_nodes, a, b, c, d):
    idx = np.searchsorted(x_nodes, x_val) - 1
    idx = max(0, min(idx, len(a) - 1))
    dx = x_val - x_nodes[idx]
    y_val = a[idx] + b[idx] * dx + c[idx] * (dx ** 2) + d[idx] * (dx ** 3)
    slope = (b[idx] + 2 * c[idx] * dx + 3 * d[idx] * (dx ** 2)) * 100
    return y_val, slope


def main():
    coords = [(48.164214, 24.536044), (48.164983, 24.534836), (48.165605, 24.534068), (48.166228, 24.532915),
              (48.166777, 24.531927), (48.167326, 24.530884), (48.167011, 24.530061), (48.166053, 24.528039),
              (48.166655, 24.526064), (48.166497, 24.523574), (48.166128, 24.520214), (48.165416, 24.517170),
              (48.164546, 24.514640), (48.163412, 24.512980), (48.162331, 24.511715), (48.162015, 24.509462),
              (48.162147, 24.506932), (48.161751, 24.504244), (48.161197, 24.501793), (48.160580, 24.500537),
              (48.160250, 24.500106)]
    elevations = np.array(
        [1263, 1285, 1285, 1332, 1308, 1317, 1317, 1339, 1375, 1417, 1487, 1524, 1553, 1629, 1756, 1794, 1828, 1886,
         1976, 1976, 2030])

    dist = [0.0]
    for i in range(1, len(coords)):
        dist.append(dist[-1] + haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1]))
    dist = np.array(dist)
    x_dense = np.linspace(dist[0], dist[-1], 500)

    a_f, b_f, c_f, d_f = get_spline_params(dist, elevations)
    res_f = [interpolate(x, dist, a_f, b_f, c_f, d_f) for x in x_dense]
    y_full = [r[0] for r in res_f]
    grads_f = [r[1] for r in res_f]

    plt.figure(1, figsize=(10, 5))
    plt.plot(dist, elevations, 'ro', label='GPS Nodes')
    plt.plot(x_dense, y_full, 'b-', label='Cubic Spline Profile')
    plt.title("Zaroslyak - Hoverla (Full Detail)")
    plt.xlabel("Distance (m)");
    plt.ylabel("Elevation (m)")
    plt.legend();
    plt.grid(True)


    plt.figure(2, figsize=(10, 5))
    plt.plot(x_dense, y_full, color='lightgray', linestyle='--', label='Full Spline (Reference)')
    for n in [10, 15]:
        idx = np.round(np.linspace(0, len(dist) - 1, n)).astype(int)
        sa, sb, sc, sd = get_spline_params(dist[idx], elevations[idx])
        plt.plot(x_dense, [interpolate(x, dist[idx], sa, sb, sc, sd)[0] for x in x_dense],
                 label=f'Spline with {n} nodes')
    plt.title("Accuracy Analysis: Impact of Node Count")
    plt.xlabel("Distance (m)");
    plt.ylabel("Elevation (m)")
    plt.legend();
    plt.grid(True)
    plt.figure(3, figsize=(10, 5))
    idx_s = np.arange(0, len(dist), 2)
    sa_s, sb_s, sc_s, sd_s = get_spline_params(dist[idx_s], elevations[idx_s])
    y_simple = np.array([interpolate(x, dist[idx_s], sa_s, sb_s, sc_s, sd_s)[0] for x in x_dense])
    error = np.abs(np.array(y_full) - y_simple)

    plt.plot(x_dense, error, 'g-', label='Error line')
    plt.fill_between(x_dense, error, color='green', alpha=0.15)
    err_nodes = np.abs(elevations - [interpolate(d, dist[idx_s], sa_s, sb_s, sc_s, sd_s)[0] for d in dist])
    plt.scatter(dist, err_nodes, color='black', s=20, label='Error at nodes')

    plt.title("Error Graph")
    plt.xlabel("Distance (m)");
    plt.ylabel("Error (m)")
    plt.legend();
    plt.grid(True)
    plt.show()

    total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
    total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, len(elevations)))

    print("\n" + "=" * 35)
    print("      ADDITIONAL TASK RESULTS")
    print("=" * 35)
    print(f"Total Distance: {dist[-1]:.2f} m")
    print(f"Total Ascent:   {total_ascent:.2f} m")
    print(f"Total Descent:  {total_descent:.2f} m")
    print(f"Max Incline:    {np.max(grads_f):.2f}%")
    print(f"Avg Gradient:   {np.mean(np.abs(grads_f)):.2f}%")
    print("-" * 35)
    work_kj = (80 * 9.81 * total_ascent) / 1000
    print(f"Mechanical Work (80kg): {work_kj:.2f} kJ")
    print(f"Energy Expenditure:     {work_kj / 4.184:.2f} kcal")
    print("=" * 35)


if __name__ == "__main__":
    main()