# visualizaciones_mejoradas.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import Normalize

#  Helpers 
def _load_history(npz_path="historial_flujo.npz"):
    data = np.load(npz_path)
    u_hist = data["u_hist"]      # shape (T, ny, nx)
    v_hist = data["v_hist"]
    Lx, Ly = float(data["Lx"]), float(data["Ly"])
    dt = float(data["dt"])
    return u_hist, v_hist, Lx, Ly, dt

def _make_grid(Lx, Ly, nx, ny):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

def _speed(u, v):
    return np.sqrt(u**2 + v**2)

#  1) Heatmap animation (u) 
def animacion_u_mejorada(npz_path="historial_flujo.npz",
                         fname="flujo_mejorado_u.mp4",
                         cmap="viridis",
                         fps=25, interval=60, alpha=0.9,
                         vmin=None, vmax=None,
                         title_prefix="Velocidad u"):
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape

    fig, ax = plt.subplots(figsize=(8, 4))
    x, y, X, Y = _make_grid(Lx, Ly, nx, ny)
    if vmin is None: vmin = np.min(u_hist)
    if vmax is None: vmax = np.max(u_hist)

    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(u_hist[0], origin="lower", extent=[0, Lx, 0, Ly],
                   aspect="auto", cmap=cmap, norm=norm, alpha=alpha)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("u")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def update(i):
        im.set_data(u_hist[i])
        ax.set_title(f"{title_prefix} — t = {(i*dt):.3f} s  (frame {i+1}/{T})")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    ani.save(fname, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Guardado: {fname}")

#  2) Partículas con cola 

def animacion_particulas_mejorada(npz_path, fname="particulas_mejorado.mp4", interval=30):

    data = np.load(npz_path)

    # Cargar historial del flujo
    U = data["u_hist"]      # (T, ny, nx)
    V = data["v_hist"]      # (T, ny, nx)

    # Parámetros del dominio
    Lx = float(data["Lx"])
    Ly = float(data["Ly"])
    nx = int(data["nx"])
    ny = int(data["ny"])

    # Reconstruir la malla
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    T = U.shape[0]  # número de snapshots

    # Crear figura
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Evolución de Partículas en el Flujo")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect("equal")

    # Inicializar partículas
    num_particles = 600
    x_particles = np.random.uniform(0, Lx, num_particles)
    y_particles = np.random.uniform(0, Ly, num_particles)

    # scatter inicial
    scat = ax.scatter(x_particles, y_particles, s=10, c=np.zeros(num_particles),
                      cmap="turbo", vmin=0, vmax=2)

    # Crear interpoladores para cada snapshot
    interpolators_u = []
    interpolators_v = []

    for t in range(T):
        interpolators_u.append(
            RegularGridInterpolator((y, x), U[t], bounds_error=False, fill_value=0.0)
        )
        interpolators_v.append(
            RegularGridInterpolator((y, x), V[t], bounds_error=False, fill_value=0.0)
        )

    def update(frame):
        nonlocal x_particles, y_particles

        interp_u = interpolators_u[frame]
        interp_v = interpolators_v[frame]

        pts = np.vstack([y_particles, x_particles]).T  # (N,2)

        u_vals = interp_u(pts)
        v_vals = interp_v(pts)

        # Avanzar partículas
        dt_local = 0.05
        x_particles += u_vals * dt_local
        y_particles += v_vals * dt_local

        # Reinsertar partículas fuera del dominio
        mask = (
            (x_particles < 0) | (x_particles > Lx) |
            (y_particles < 0) | (y_particles > Ly)
        )
        x_particles[mask] = np.random.uniform(0, Lx, mask.sum())
        y_particles[mask] = np.random.uniform(0, Ly, mask.sum())

        # actualizar colores
        speeds = np.sqrt(u_vals**2 + v_vals**2)
        scat.set_offsets(np.column_stack([x_particles, y_particles]))
        scat.set_array(speeds)

        return scat,

    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=interval, blit=True)

    ani.save(fname, writer="ffmpeg")
    print(f"Guardado: {fname}")

    plt.close(fig)


#  3) Streamlines animadas 
def animacion_streamlines(npz_path="historial_flujo.npz",
                          fname="streamlines.mp4",
                          density=1.5,  # densidad de streamlines
                          cmap="plasma", fps=25, interval=80):
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # background speed colormap
    im = ax.imshow(_speed(u_hist[0], v_hist[0]), origin="lower", extent=[0, Lx, 0, Ly],
                   aspect="auto", cmap="viridis", alpha=0.6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("speed")

    streams = None
    def update(frame):
        ax.clear()  # limpia streamlines previas
        u = u_hist[frame]
        v = v_hist[frame]
        speed = _speed(u, v)
        im.set_data(speed)
        # streamplot: density controla cuantas líneas
        strm = ax.streamplot(x, y, u, v, density=density, linewidth=1, arrowsize=1)
        ax.set_title(f"Streamlines — t = {(frame*dt):.3f} s (frame {frame+1}/{T})")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    ani.save(fname, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Guardado: {fname}")

#  4) Vorticidad 
def animacion_vorticidad(npz_path="historial_flujo.npz",
                         fname="vorticidad.mp4",
                         cmap="coolwarm", fps=25, interval=80, vmin=None, vmax=None):
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx = x[1] - x[0]; dy = y[1] - y[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    im = ax.imshow(np.zeros((ny, nx)), origin="lower", extent=[0, Lx, 0, Ly],
                   aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("vorticidad (ω)")

    if vmin is None or vmax is None:
        # estimate from first frame
        u0, v0 = u_hist[0], v_hist[0]
        w0 = np.gradient(v0, dx, axis=1) - np.gradient(u0, dy, axis=0)
        if vmin is None: vmin = np.min(w0)
        if vmax is None: vmax = np.max(w0)
    im.set_clim(vmin, vmax)

    def update(frame):
        u = u_hist[frame]
        v = v_hist[frame]
        # ω = dv/dx - du/dy
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        w = dvdx - dudy
        im.set_data(w)
        ax.set_title(f"Vorticidad — t = {(frame*dt):.3f} s (frame {frame+1}/{T})")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    ani.save(fname, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Guardado: {fname}")

#  5) Quiver snapshot (vector field) 
def quiver_snapshot(frame_index=0, npz_path="historial_flujo.npz",
                    fname="quiver_snapshot.png", stride=6, scale=1.0):
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape
    frame_index = np.clip(frame_index, 0, T-1)
    u = u_hist[frame_index]; v = v_hist[frame_index]
    x = np.linspace(0, Lx, nx); y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.imshow(_speed(u, v), origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis", alpha=0.5)
    Q = ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                  u[::stride, ::stride], v[::stride, ::stride],
                  pivot='mid', scale=None)
    ax.set_title(f"Quiver frame {frame_index} (stride={stride})")
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"Guardado: {fname}")
