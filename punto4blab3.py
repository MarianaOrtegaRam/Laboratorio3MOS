import numpy as np
import matplotlib.pyplot as plt

# Definición de la función objetivo y su gradiente
f = lambda x, y: (x - 2)**2 * (y + 2)**2 + (x + 1)**2 + (y - 1)**2

def gradiente(x, y):
    df_dx = 2 * (x - 2) * (y + 2)**2 + 2 * (x + 1)
    df_dy = 2 * (x - 2)**2 * (y + 2) + 2 * (y - 1)
    return np.array([df_dx, df_dy])

# Matriz Hessiana
def hessiana(x, y):
    d2f_dx2 = 2 * (y + 2)**2 + 2
    d2f_dy2 = 2 * (x - 2)**2 + 2
    d2f_dxdy = 4 * (x - 2) * (y + 2)
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

# Parámetros iniciales y configuración
x0, y0 = -2, -3
alfa = 0.01
max_iter = 100

# Gradiente Descendente
x, y = x0, y0
trayectoria_gd = [(x, y)]
for _ in range(max_iter):
    grad = gradiente(x, y)
    x -= alfa * grad[0]
    y -= alfa * grad[1]
    trayectoria_gd.append((x, y))

# Newton-Raphson
x, y = x0, y0
trayectoria_nr = [(x, y)]
for _ in range(max_iter):
    grad = gradiente(x, y)
    hess = hessiana(x, y)
    delta = np.linalg.solve(hess, grad)  # Resolver H * delta = grad
    x -= delta[0]
    y -= delta[1]
    trayectoria_nr.append((x, y))

# Gráfica
x_vals = np.linspace(-4, 4, 400)
y_vals = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.figure(figsize=(12, 8))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
trayectoria_gd = np.array(trayectoria_gd)
trayectoria_nr = np.array(trayectoria_nr)

plt.plot(trayectoria_gd[:, 0], trayectoria_gd[:, 1], marker='o', color='red', markersize=3, label='Gradiente Descendente')
plt.plot(trayectoria_nr[:, 0], trayectoria_nr[:, 1], marker='x', color='blue', markersize=3, label='Newton-Raphson')
plt.scatter(2, -1, color='green', marker='*', s=100, label='Mínimo Global (2, -1)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación: Newton-Raphson vs Gradiente Descendente')
plt.legend()
plt.grid(True)
plt.show()
