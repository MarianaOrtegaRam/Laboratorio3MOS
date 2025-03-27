import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rosenbrock
def f(x, y):
    return (x - 1)**2 + 100 * (y - x**2)**2

# Gradiente
def grad_f(x, y):
    dfdx = 2 * (x - 1) - 400 * x * (y - x**2)
    dfdy = 200 * (y - x**2)
    return np.array([dfdx, dfdy])

# Hessiana
def hessian(x, y):
    d2fdx2 = 2 - 400 * y + 1200 * x**2
    d2fdxdy = -400 * x
    d2fdy2 = 200
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

# Newton-Raphson
def newton_raphson(x0, y0, tol=1e-6, max_iter=100):
    x, y = x0, y0
    history = [(x, y)]

    for _ in range(max_iter):
        grad = grad_f(x, y)
        H = hessian(x, y)
        try:
            inv_H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("La matriz Hessiana no es invertible.")
            break

        delta = -inv_H @ grad
        x, y = x + delta[0], y + delta[1]
        history.append((x, y))

        if np.linalg.norm(delta) < tol:
            print(f"Convergencia alcanzada en {_ + 1} iteraciones.")
            break

    return np.array(history)

# Parámetros
x0, y0 = 0, 10
historial = newton_raphson(x0, y0)
print("Historial de puntos:")
print(historial)

# Graficar
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Superficie
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Puntos iterativos en rojo y punto final resaltado
ax.scatter(historial[:, 0], historial[:, 1], f(historial[:, 0], historial[:, 1]), color='red', s=50, label="Iteraciones")
ax.scatter(1, 1, f(1, 1), color='black', s=80, label="Mínimo conocido (1,1)", marker='x')

ax.set_title("Superficie de la función de Rosenbrock y puntos iterativos")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.legend()
plt.show()
