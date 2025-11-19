import numpy as np

data = np.load("historial_flujo.npz")

u_hist = data["u_hist"]   # array de snapshots de u
v_hist = data["v_hist"]   # array de snapshots de v

print("Tamaño de u_hist:", u_hist.shape)  # (num_snapshots, ny, nx)
print("Velocidad máxima en la simulación:", np.max(u_hist))
