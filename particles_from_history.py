import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RectBivariateSpline

def animate_from_history():
    data = np.load("historial_flujo.npz")
    u_hist = data["u_hist"]
    v_hist = data["v_hist"]
    Lx, Ly = float(data["Lx"]), float(data["Ly"])
    ny, nx = u_hist.shape[1:]
    dt = float(data["dt"])

    n_particles = 300
    x_particles = np.random.uniform(0.05, 0.2, n_particles)
    y_particles = np.random.uniform(0.3, 0.7, n_particles)
    colors = np.ones(n_particles) * 0.3

    x_grid = np.linspace(0, Lx, nx)
    y_grid = np.linspace(0, Ly, ny)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    img = ax.imshow(u_hist[0], origin="lower", extent=[0, Lx, 0, Ly], cmap="turbo", alpha=0.3)
    scat = ax.scatter(x_particles, y_particles, s=10, c=colors, cmap="plasma", vmin=0, vmax=1)
    ax.set_title("Evolución de partículas en flujo precomputado")

    def update(frame):
        u = u_hist[frame]
        v = v_hist[frame]
        interp_u = RectBivariateSpline(y_grid, x_grid, u)
        interp_v = RectBivariateSpline(y_grid, x_grid, v)

        for i in range(len(x_particles)):
            ux = interp_u(y_particles[i], x_particles[i])[0, 0]
            vy = interp_v(y_particles[i], x_particles[i])[0, 0]
            x_particles[i] += ux * dt * 2
            y_particles[i] += vy * dt * 2
            if x_particles[i] > Lx:
                x_particles[i] = 0.0
                y_particles[i] = np.random.uniform(0.3, 0.7)
            if y_particles[i] < 0 or y_particles[i] > Ly:
                y_particles[i] = np.random.uniform(0.3, 0.7)
            colors[i] = min(1.0, np.sqrt(ux**2 + vy**2) * 10)  # color según velocidad
        scat.set_offsets(np.column_stack((x_particles, y_particles)))
        scat.set_array(colors)
        img.set_data(u)
        ax.set_title(f"Frame {frame+1}/{len(u_hist)}")
        return [img, scat]

    ani = animation.FuncAnimation(fig, update, frames=len(u_hist), interval=80, blit=True)
    ani.save("particulas_history.mp4", writer="ffmpeg", fps=30, dpi=150)
    plt.close(fig)
    print("🎞️ Animación guardada como particulas_history.mp4")

if __name__ == "__main__":
    animate_from_history()
