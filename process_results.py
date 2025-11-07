import numpy as np
data = np.load("resultados.npz")

u = data["u"]
v = data["v"]
p = data["p"]

print("Tamaño de u:", u.shape)
print("Velocidad máxima:", np.max(u))
