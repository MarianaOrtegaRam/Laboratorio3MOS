import numpy as np
import matplotlib.pyplot as plt

# Función objetivo
def funcion_objetivo(x, y, z):
    return (x - 1)**2 + (y - 2)**2 + (z - 3)**2

# Gradiente
def gradiente(x, y, z):
    return np.array([2 * (x - 1), 2 * (y - 2), 2 * (z - 3)])

# Matriz Hessiana
def hessiana(x, y, z):
    return np.array([[2, 0, 0],
                     [0, 2, 0],
                     [0, 0, 2]])

# Newton-Raphson
def newton_raphson(x0, y0, z0, tol=1e-6, max_iter=100):
    iteracion = 0
    punto = np.array([x0, y0, z0])
    trayectoria = [punto.copy()]
    while iteracion < max_iter:
        grad = gradiente(*punto)
        if np.linalg.norm(grad) < tol:
            break
        hess_inv = np.linalg.inv(hessiana(*punto))
        punto = punto - np.dot(hess_inv, grad)
        trayectoria.append(punto.copy())
        iteracion += 1

    return punto, iteracion, np.array(trayectoria)

# Ejemplo de uso
resultado, iteraciones, trayectoria = newton_raphson(0, 0, 0)

# Graficar
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(trayectoria[:, 0], trayectoria[:, 1], 'o-', color='blue')
ax1.set_title('Proyección en el plano XY')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.plot(trayectoria[:, 0], trayectoria[:, 2], 'o-', color='green')
ax2.set_title('Proyección en el plano XZ')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')

ax3.plot(trayectoria[:, 1], trayectoria[:, 2], 'o-', color='red')
ax3.set_title('Proyección en el plano YZ')
ax3.set_xlabel('Y')
ax3.set_ylabel('Z')

plt.tight_layout()
plt.show()

print(f"Mínimo encontrado en: {resultado} en {iteraciones} iteraciones")
