import numpy as np
import matplotlib.pyplot as plt

# Definición de la función objetivo, gradiente y Hessiana
f = lambda x, y: (x - 2)**2 * (y + 2)**2 + (x + 1)**2 + (y - 1)**2
gradiente = lambda x, y: np.array([2 * (x - 2) * (y + 2)**2 + 2 * (x + 1), 2 * (x - 2)**2 * (y + 2) + 2 * (y - 1)])
Hessiana = lambda x, y: np.array([
    [2 * (y + 2)**2 + 2, 4 * (x - 2) * (y + 2)],
    [4 * (x - 2) * (y + 2), 2 * (x - 2)**2 + 2]
])

# Parámetros iniciales
x0, y0 = -2, -3
max_iter = 50

# Valores de alpha para experimentar
alphas = [0.01, 0.05, 0.1, 0.5, 1.0]

# Gradiente Descendente
plt.figure(figsize=(12, 6))
x_vals = np.linspace(-4, 4, 400)
y_vals = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)
plt.contour(X, Y, Z, levels=50, cmap='viridis')

for alpha in alphas:
    x, y = x0, y0
    trayectoria = [(x, y)]
    for _ in range(max_iter):
        grad = gradiente(x, y)
        x -= alpha * grad[0]
        y -= alpha * grad[1]
        trayectoria.append((x, y))
    trayectoria = np.array(trayectoria)
    plt.plot(trayectoria[:, 0], trayectoria[:, 1], marker='o', label=f'GD α={alpha}', markersize=3)

# Newton-Raphson
x, y = x0, y0
trayectoria_nr = [(x, y)]
for _ in range(max_iter):
    grad = gradiente(x, y)
    H_inv = np.linalg.inv(Hessiana(x, y))
    delta = H_inv @ grad
    x -= delta[0]
    y -= delta[1]
    trayectoria_nr.append((x, y))
trayectoria_nr = np.array(trayectoria_nr)
plt.plot(trayectoria_nr[:, 0], trayectoria_nr[:, 1], marker='x', label='Newton-Raphson', markersize=4, color='red')

plt.scatter(2, -1, color='blue', marker='x', s=100, label='Mínimo Global (2, -1)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación de Trayectorias para Diferentes Alphas y Newton-Raphson')
plt.legend()
plt.grid(True)
plt.show()
