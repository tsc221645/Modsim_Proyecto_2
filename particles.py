import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RectBivariateSpline
from solver import pressure_poisson, step_navier_stokes, forcing_field
from physics import build_rhs
from utils import default_params

def animate_particles():
    params = default_params()
    nx, ny = params["nx"], params["ny"]
    Lx, Ly = params["Lx"], params["Ly"]
    dx, dy = params["dx"], params["dy"]
    dt, nt, nit = params["dt"], params["nt"], params["nit"]
    rho, nu = params["rho"], params["nu"]

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Posiciones iniciales de partículas (una nube en el lado izquierdo)
    n_particles = 400
    x_particles = np.random.uniform(0.05, 0.2, n_particles)
    y_particles = np.random.uniform(0.3, 0.7, n_particles)

    # Preparar figura
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    scat = ax.scatter(x_particles, y_particles, s=8, c='white')
    img = ax.imshow(u, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="turbo", alpha=0.4)
    ax.set_title("Ingreso de partículas en el flujo")

    # Interpoladores para u,v
    x_grid = np.linspace(0, Lx, nx)
    y_grid = np.linspace(0, Ly, ny)
    interp_u = RectBivariateSpline(y_grid, x_grid, u)
    interp_v = RectBivariateSpline(y_grid, x_grid, v)

    sample_every = 50
    def update(frame):
        nonlocal u, v, p, interp_u, interp_v, x_particles, y_particles

        # Resolver fluido unos cuantos pasos antes de cada frame
        for _ in range(sample_every):
            b = build_rhs(u, v, dx, dy, dt, rho)
            p = pressure_poisson(p, b, dx, dy, nit)
            Fx_field, Fy_field = forcing_field(params, frame*dt, ny, nx, dx, dy)
            u, v = step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx_field, Fy_field)

        # Actualizar interpoladores
        interp_u = RectBivariateSpline(y_grid, x_grid, u)
        interp_v = RectBivariateSpline(y_grid, x_grid, v)

        # Mover partículas
        for i in range(len(x_particles)):
            ux = interp_u(y_particles[i], x_particles[i])[0, 0]
            vy = interp_v(y_particles[i], x_particles[i])[0, 0]
            x_particles[i] += ux * dt
            y_particles[i] += vy * dt

            # Reingresar si salen del dominio
            if x_particles[i] > Lx:
                x_particles[i] = 0.0
                y_particles[i] = np.random.uniform(0.3, 0.7)
            if y_particles[i] < 0 or y_particles[i] > Ly:
                y_particles[i] = np.random.uniform(0.3, 0.7)

        scat.set_offsets(np.column_stack((x_particles, y_particles)))
        img.set_data(u)
        ax.set_title(f"t = {frame * sample_every * dt:.3f} s")
        return [img, scat]

    ani = animation.FuncAnimation(fig, update, frames=150, interval=80, blit=True)
    ani.save("particulas_flujo.mp4", writer="ffmpeg", fps=30, dpi=150)
    plt.close(fig)
    print("🎞️ Animación de partículas guardada como particulas_flujo.mp4")

if __name__ == "__main__":
    animate_particles()
