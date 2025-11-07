import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from solver import pressure_poisson, step_navier_stokes, forcing_field
from physics import build_rhs
from utils import default_params

def animate_flow(save_mp4=True, save_gif=False):
    params = default_params()
    nx, ny = params["nx"], params["ny"]
    Lx, Ly = params["Lx"], params["Ly"]
    dx, dy = params["dx"], params["dy"]
    dt, nt, nit = params["dt"], params["nt"], params["nit"]
    rho, nu = params["rho"], params["nu"]

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    fig, ax = plt.subplots(figsize=(6, 3))
    img = ax.imshow(u, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="turbo")
    plt.colorbar(img, ax=ax)
    ax.set_title("Evolución de la velocidad u")

    sample_every = 20
    snapshots = []
    for n in range(1, nt + 1):
        t = n * dt
        b = build_rhs(u, v, dx, dy, dt, rho)
        p = pressure_poisson(p, b, dx, dy, nit)
        Fx_field, Fy_field = forcing_field(params, t, ny, nx, dx, dy)
        u, v = step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx_field, Fy_field)
        if n % sample_every == 0:
            snapshots.append(u.copy())
            print(f"Frame {len(snapshots)} guardado")

    def update(i):
        img.set_data(snapshots[i])
        ax.set_title(f"t = {i * sample_every * dt:.3f} s")
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=50, blit=True)
    if save_mp4:
        ani.save("flujo_animacion.mp4", writer="ffmpeg", fps=30, dpi=150)
    if save_gif:
        ani.save("flujo_animacion.gif", writer="imagemagick", fps=20)
    plt.close(fig)
    print("🎞️ Animación creada correctamente.")
    return ani

if __name__ == "__main__":
    animate_flow()
