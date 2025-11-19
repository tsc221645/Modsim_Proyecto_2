# visualizaciones_mejoradas.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import Normalize

# ============================================================================
# FUNCIONES AUXILIARES (Helpers)
# ============================================================================

def _load_history(npz_path="historial_flujo.npz"):
    """
    Carga los datos de simulación guardados en formato NPZ.
    
    Returns:
        u_hist, v_hist: Historial de velocidades (T, ny, nx)
        Lx, Ly: Dimensiones del dominio
        dt: Paso de tiempo
    """
    data = np.load(npz_path)
    u_hist = data["u_hist"]      # shape (T, ny, nx) - T frames temporales
    v_hist = data["v_hist"]
    Lx, Ly = float(data["Lx"]), float(data["Ly"])
    dt = float(data["dt"])
    return u_hist, v_hist, Lx, Ly, dt

def _make_grid(Lx, Ly, nx, ny):
    """
    Crea una malla de coordenadas para visualización.
    """
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)  # Mallas 2D para coordenadas
    return x, y, X, Y

def _speed(u, v):
    """
    Calcula la magnitud de la velocidad: |u| = √(u² + v²)
    """
    return np.sqrt(u**2 + v**2)

# ============================================================================
# 1) ANIMACIÓN DE MAPA DE CALOR (Campo de velocidad u)
# ============================================================================

def animacion_u_mejorada(npz_path="historial_flujo.npz",
                         fname="flujo_mejorado_u.mp4",
                         cmap="viridis",
                         fps=25, interval=60, alpha=0.9,
                         vmin=None, vmax=None,
                         title_prefix="Velocidad u"):
    """
    Crea una animación del campo de velocidad u con escala de colores constante.
    
    Args:
        npz_path: Ruta al archivo de datos
        fname: Nombre del archivo de salida
        cmap: Mapa de colores
        fps: Frames por segundo del video
        interval: Intervalo entre frames en ms
        alpha: Transparencia
        vmin, vmax: Límites de la escala de colores
        title_prefix: Título de la animación
    """
    # Cargar datos
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape  # T = número de frames temporales

    # Configurar figura
    fig, ax = plt.subplots(figsize=(8, 4))
    x, y, X, Y = _make_grid(Lx, Ly, nx, ny)
    
    # Establecer límites de color si no se especifican
    if vmin is None: vmin = np.min(u_hist)
    if vmax is None: vmax = np.max(u_hist)

    # Crear normalizador de colores y imagen inicial
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(u_hist[0], origin="lower", extent=[0, Lx, 0, Ly],
                   aspect="auto", cmap=cmap, norm=norm, alpha=alpha)
    
    # Barra de color
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("u")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def update(i):
        """Función de actualización para cada frame de la animación"""
        im.set_data(u_hist[i])  # Actualizar datos
        ax.set_title(f"{title_prefix} — t = {(i*dt):.3f} s  (frame {i+1}/{T})")
        return [im]

    # Crear y guardar animación
    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    ani.save(fname, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Guardado: {fname}")

# ============================================================================
# 2) ANIMACIÓN DE PARTÍCULAS CON INTERPOLACIÓN MEJORADA
# ============================================================================

def animacion_particulas_mejorada(npz_path, fname="particulas_mejorado.mp4", interval=30):
    """
    Crea una animación de partículas que se mueven con el flujo usando interpolación.
    
    Características:
    - Interpolación suave entre puntos de malla
    - Reinicio de partículas que salen del dominio
    - Color según velocidad local
    """
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
    ax.set_aspect("equal")  # Mantener relación de aspecto

    # Inicializar partículas (distribución aleatoria uniforme)
    num_particles = 600
    x_particles = np.random.uniform(0, Lx, num_particles)
    y_particles = np.random.uniform(0, Ly, num_particles)

    # Scatter plot inicial (partículas)
    scat = ax.scatter(x_particles, y_particles, s=10, c=np.zeros(num_particles),
                      cmap="turbo", vmin=0, vmax=2)

    # Crear interpoladores para cada snapshot temporal
    # RegularGridInterpolator permite interpolación suave en coordenadas arbitrarias
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

        # Obtener interpoladores para el frame actual
        interp_u = interpolators_u[frame]
        interp_v = interpolators_v[frame]

        # Preparar puntos para interpolación (formato: [y, x])
        pts = np.vstack([y_particles, x_particles]).T  # (N,2)

        # Interpolar velocidades en posiciones de partículas
        u_vals = interp_u(pts)
        v_vals = interp_v(pts)

        # Avanzar partículas usando método de Euler
        dt_local = 0.05  # Paso de tiempo para integración de trayectorias
        x_particles += u_vals * dt_local
        y_particles += v_vals * dt_local

        # Reinsertar partículas que salen del dominio
        mask = (
            (x_particles < 0) | (x_particles > Lx) |
            (y_particles < 0) | (y_particles > Ly)
        )
        x_particles[mask] = np.random.uniform(0, Lx, mask.sum())
        y_particles[mask] = np.random.uniform(0, Ly, mask.sum())

        # Actualizar colores según velocidad
        speeds = np.sqrt(u_vals**2 + v_vals**2)
        scat.set_offsets(np.column_stack([x_particles, y_particles]))
        scat.set_array(speeds)

        return scat,

    # Crear animación
    ani = animation.FuncAnimation(fig, update, frames=T,
                                  interval=interval, blit=True)

    ani.save(fname, writer="ffmpeg")
    print(f"Guardado: {fname}")
    plt.close(fig)

# ============================================================================
# 3) ANIMACIÓN DE STREAMLINES (Líneas de corriente)
# ============================================================================

def animacion_streamlines(npz_path="historial_flujo.npz",
                          fname="streamlines.mp4",
                          density=1.5,  # densidad de streamlines
                          cmap="plasma", fps=25, interval=80):
    """
    Crea una animación de líneas de corriente sobre un mapa de velocidad.
    
    Las streamlines muestran la dirección instantánea del flujo.
    """
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Fondo: mapa de colores de velocidad
    im = ax.imshow(_speed(u_hist[0], v_hist[0]), origin="lower", extent=[0, Lx, 0, Ly],
                   aspect="auto", cmap="viridis", alpha=0.6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("speed")

    def update(frame):
        ax.clear()  # Limpiar streamlines previas
        u = u_hist[frame]
        v = v_hist[frame]
        speed = _speed(u, v)
        im.set_data(speed)
        
        # Crear nuevas streamlines
        # density controla cuántas líneas se dibujan
        strm = ax.streamplot(x, y, u, v, density=density, linewidth=1, arrowsize=1)
        
        ax.set_title(f"Streamlines — t = {(frame*dt):.3f} s (frame {frame+1}/{T})")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    ani.save(fname, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Guardado: {fname}")

# ============================================================================
# 4) ANIMACIÓN DE VORTICIDAD
# ============================================================================

def animacion_vorticidad(npz_path="historial_flujo.npz",
                         fname="vorticidad.mp4",
                         cmap="coolwarm", fps=25, interval=80, vmin=None, vmax=None):
    """
    Crea una animación del campo de vorticidad ω = ∇ × u = dv/dx - du/dy.
    
    La vorticidad mide la rotación local del fluido y es clave para identificar
    estructuras vorticiales.
    """
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx = x[1] - x[0]; dy = y[1] - y[0]  # Calcula diferenciales

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    
    # Imagen inicial (ceros)
    im = ax.imshow(np.zeros((ny, nx)), origin="lower", extent=[0, Lx, 0, Ly],
                   aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("vorticidad (ω)")

    # Establecer límites de color si no se especifican
    if vmin is None or vmax is None:
        # Calcular del primer frame como referencia
        u0, v0 = u_hist[0], v_hist[0]
        w0 = np.gradient(v0, dx, axis=1) - np.gradient(u0, dy, axis=0)
        if vmin is None: vmin = np.min(w0)
        if vmax is None: vmax = np.max(w0)
    im.set_clim(vmin, vmax)

    def update(frame):
        u = u_hist[frame]
        v = v_hist[frame]
        
        # Calcular vorticidad: ω = dv/dx - du/dy
        dvdx = np.gradient(v, dx, axis=1)  # Derivada en x de v
        dudy = np.gradient(u, dy, axis=0)  # Derivada en y de u
        w = dvdx - dudy
        
        im.set_data(w)
        ax.set_title(f"Vorticidad — t = {(frame*dt):.3f} s (frame {frame+1}/{T})")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    ani.save(fname, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Guardado: {fname}")

# ============================================================================
# 5) SNAPSHOT DE CAMPO VECTORIAL (Quiver plot)
# ============================================================================

def quiver_snapshot(frame_index=0, npz_path="historial_flujo.npz",
                    fname="quiver_snapshot.png", stride=6, scale=1.0):
    """
    Crea un snapshot estático del campo vectorial en un frame específico.
    
    Args:
        frame_index: Frame temporal a visualizar
        stride: Salto entre vectores (reduce densidad)
        scale: Escala de los vectores
    """
    u_hist, v_hist, Lx, Ly, dt = _load_history(npz_path)
    T, ny, nx = u_hist.shape
    frame_index = np.clip(frame_index, 0, T-1)  # Asegurar índice válido
    
    u = u_hist[frame_index]; v = v_hist[frame_index]
    x = np.linspace(0, Lx, nx); y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    
    # Fondo: magnitud de velocidad
    ax.imshow(_speed(u, v), origin="lower", extent=[0, Lx, 0, Ly], cmap="viridis", alpha=0.5)
    
    # Campo vectorial (submuestreado para claridad)
    Q = ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                  u[::stride, ::stride], v[::stride, ::stride],
                  pivot='mid', scale=None)  # scale=None ajusta automáticamente
    
    ax.set_title(f"Quiver frame {frame_index} (stride={stride})")
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"Guardado: {fname}")