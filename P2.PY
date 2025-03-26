import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Definición de la función y derivadas
# ---------------------------

def f(x):
    return x**5 - 8*x**3 + 10*x + 6

def df(x):
    return 5*x**4 - 24*x**2 + 10

def d2f(x):
    return 20*x**3 - 48*x

# ---------------------------
# 2. Implementación del algoritmo Newton-Raphson para extremos
# ---------------------------

def newton_raphson(fprime, fdoubleprime, x0, alpha=1.0, tol=1e-6, max_iter=100):
    xk = x0
    trajectory = [xk]
    for _ in range(max_iter):
        f1 = fprime(xk)
        f2 = fdoubleprime(xk)
        if abs(f1) < tol:
            break
        if f2 == 0:
            break
        xk = xk - alpha * (f1 / f2)
        trajectory.append(xk)
    return xk, trajectory

# ---------------------------
# 3. Búsqueda de extremos
# ---------------------------

alpha = 0.6
x0_vals = np.linspace(-3, 3, 25)
extremos = []
traj_all = []

for x0 in x0_vals:
    x_star, traj = newton_raphson(df, d2f, x0, alpha)
    x_star = np.round(x_star, 5)  # redondeo para evitar duplicados numéricos
    if abs(df(x_star)) < 1e-5 and not any(abs(x_star - x[0]) < 1e-3 for x in extremos):
        tipo = 'mínimo' if d2f(x_star) > 0 else 'máximo'
        extremos.append((x_star, f(x_star), tipo))
    traj_all.append((x0, traj))

# ---------------------------
# 4. Clasificación de extremos globales
# ---------------------------

if extremos:
    min_global = min(extremos, key=lambda x: x[1])
    max_global = max(extremos, key=lambda x: x[1])
else:
    min_global = max_global = None

# ---------------------------
# 5. Visualización
# ---------------------------

x = np.linspace(-3, 3, 500)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x)', color='blue')
for x_, y_, tipo in extremos:
    color = 'black'
    marker = 'o'
    if (x_, y_, tipo) == min_global or (x_, y_, tipo) == max_global:
        color = 'red'
    plt.scatter(x_, y_, color=color, label=f'{tipo} en x={x_:.2f}', marker=marker)

plt.title("Extremos locales y globales con Newton-Raphson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# 6. Reporte textual
# ---------------------------

print("Extremos encontrados:")
for x_, y_, tipo in extremos:
    glb = ''
    if (x_, y_, tipo) == min_global:
        glb = '(mínimo global)'
    elif (x_, y_, tipo) == max_global:
        glb = '(máximo global)'
    print(f"{tipo.title()} en x = {x_}, f(x) = {y_:.4f} {glb}")
