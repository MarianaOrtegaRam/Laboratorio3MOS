import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    Funcion polinómica de tercer grado.
    f(x) = 3x^3 - 10x^2 - 56x + 50
    """
    return 3*x**3 - 10*x**2 - 56*x + 50

def df(x):
    """
    Primera derivada de f(x):
    f'(x) = 9x^2 - 20x - 56
    """
    return 9*x**2 - 20*x - 56

def d2f(x):
    """
    Segunda derivada de f(x):
    f''(x) = 18x - 20
    """
    return 18*x - 20

# ---------------------------
# 2. Implementación del algoritmo de Newton-Raphson
# ---------------------------

def newton_raphson(fprime, fdoubleprime, x0, alpha=1.0, tol=1e-6, max_iter=100):
    """
    Implementa el método de Newton-Raphson para encontrar extremos locales.
    
    Parámetros:
    fprime -- primera derivada de f
    fdoubleprime -- segunda derivada de f
    x0 -- valor inicial
    alpha -- factor de convergencia
    tol -- tolerancia de parada
    max_iter -- número máximo de iteraciones
    
    Retorna:
    xk -- punto al que converge
    trayectoria -- lista de valores xk en cada iteración
    """
    xk = x0
    trayectoria = [xk]
    for _ in range(max_iter):
        f1 = fprime(xk)
        f2 = fdoubleprime(xk)
        if abs(f1) < tol:
            break
        if f2 == 0:
            print("Advertencia: Segunda derivada nula. No se puede continuar.")
            break
        xk = xk - alpha * (f1 / f2)
        trayectoria.append(xk)
    return xk, trayectoria

# ---------------------------
# 3. Experimentación con valores iniciales y alpha
# ---------------------------

alpha = 0.6
x0_vals = np.linspace(-6, 6, 25)
minimos = []
maximos = []
traj_all = []

for x0 in x0_vals:
    x_star, traj = newton_raphson(df, d2f, x0, alpha=alpha)
    fpp = d2f(x_star)
    if abs(df(x_star)) < 1e-5:
        if fpp > 0:
            minimos.append((x_star, f(x_star)))
        elif fpp < 0:
            maximos.append((x_star, f(x_star)))
    traj_all.append((x0, traj))

# ---------------------------
# 4. Visualización gráfica de resultados
# ---------------------------

x = np.linspace(-6, 6, 400)
y = f(x)

plt.figure(figsize=(10,6))
plt.plot(x, y, label='f(x)', color='blue')
if minimos:
    plt.scatter(*zip(*minimos), color='green', label='Mínimos locales', marker='o')
if maximos:
    plt.scatter(*zip(*maximos), color='red', label='Máximos locales', marker='x')
plt.title("Extremos locales encontrados con Newton-Raphson")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# 5. Análisis de resultados
# ---------------------------

print("Resumen de resultados:\n")
print("Puntos críticos encontrados:")
for x, y in minimos:
    print(f"Mínimo local en x = {x:.4f}, f(x) = {y:.4f}")
for x, y in maximos:
    print(f"Máximo local en x = {x:.4f}, f(x) = {y:.4f}")

print("\nTrayectorias de convergencia desde valores iniciales seleccionados:")
for x0, traj in traj_all:
    print(f"x0 = {x0:.2f} → x* ≈ {traj[-1]:.4f} en {len(traj)} iteraciones")
