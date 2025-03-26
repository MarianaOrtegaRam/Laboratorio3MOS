import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# 1. Definición de la función y derivadas parciales
# ---------------------------

def f(x, y):
    return x**4 + y**4 - 4*x*y + 1

def grad_f(x, y):
    dfdx = 4*x**3 - 4*y
    dfdy = 4*y**3 - 4*x
    return np.array([dfdx, dfdy])

def hessian_f(x, y):
    d2fdx2 = 12*x**2
    d2fdy2 = 12*y**2
    d2fdxdy = -4
    return np.array([[d2fdx2, d2fdxdy],
                     [d2fdxdy, d2fdy2]])

# ---------------------------
# 2. Implementación del método de Newton-Raphson 2D
# ---------------------------

def newton_raphson_2d(x0, alpha=1.0, tol=1e-6, max_iter=100):
    xk = np.array(x0, dtype=float)
    trajectory = [xk.copy()]
    for _ in range(max_iter):
        grad = grad_f(xk[0], xk[1])
        hess = hessian_f(xk[0], xk[1])
        if np.linalg.norm(grad) < tol:
            break
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Matriz Hessiana singular")
            break
        xk = xk - alpha * delta
        trajectory.append(xk.copy())
    return xk, trajectory

# ---------------------------
# 3. Pruebas con distintos puntos iniciales
# ---------------------------

x0_list = [(-2, -2), (-1, 1), (2, 2), (0.5, -0.5), (0, 0)]
resultados = []

for x0 in x0_list:
    punto, trayectoria = newton_raphson_2d(x0, alpha=1.0)
    resultados.append((x0, punto, f(punto[0], punto[1]), trayectoria))

# ---------------------------
# 4. Visualización en 3D
# ---------------------------

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
X = Y = np.linspace(-2.5, 2.5, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

for _, punto, _, _ in resultados:
    ax.scatter(punto[0], punto[1], f(punto[0], punto[1]), color='red', s=50)

ax.set_title("Extremos de f(x, y) con Newton-Raphson 2D")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.show()

# ---------------------------
# 5. Resultados textuales
# ---------------------------
print("\nResultados del método Newton-Raphson 2D:\n")
for x0, punto, valor, _ in resultados:
    print(f"Desde x0 = {x0} → Converge a ({punto[0]:.4f}, {punto[1]:.4f}) con f = {valor:.4f}")
