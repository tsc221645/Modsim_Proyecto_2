import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RectBivariateSpline

def animate_from_history():
    # Cargar los datos de la simulación guardados previamente
    data = np.load("historial_flujo.npz")
    u_hist = data["u_hist"]  # historial de velocidad en x
    v_hist = data["v_hist"]  # historial de velocidad en y
    Lx, Ly = float(data["Lx"]), float(data["Ly"])  # dimensiones del dominio
    ny, nx = u_hist.shape[1:]  # dimensiones de la malla
    dt = float(data["dt"])     # paso de tiempo

    # Inicializar partículas para visualización
    n_particles = 300  # número de partículas
    # Posicionar partículas aleatoriamente en una región específica (zona de chorro)
    x_particles = np.random.uniform(0.05, 0.2, n_particles)  # coordenadas x iniciales
    y_particles = np.random.uniform(0.3, 0.7, n_particles)  # coordenadas y iniciales
    colors = np.ones(n_particles) * 0.3  # colores iniciales de las partículas

    # Crear malla para interpolación
    x_grid = np.linspace(0, Lx, nx)  # puntos x de la malla
    y_grid = np.linspace(0, Ly, ny)  # puntos y de la malla

    # Configurar figura y ejes para la animación
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)  # establecer límites del dominio
    
    # Crear imagen de fondo que muestra el campo de velocidad u
    img = ax.imshow(u_hist[0], origin="lower", extent=[0, Lx, 0, Ly], 
                    cmap="turbo", alpha=0.3)
    
    # Crear scatter plot para las partículas
    scat = ax.scatter(x_particles, y_particles, s=10, c=colors, 
                      cmap="plasma", vmin=0, vmax=1)
    ax.set_title("Evolución de partículas en flujo precomputado")

    # Función que se ejecuta para cada frame de la animación
    def update(frame):
        # Obtener campos de velocidad para el frame actual
        u = u_hist[frame]
        v = v_hist[frame]
        
        # Crear funciones de interpolación para obtener velocidades en cualquier punto
        interp_u = RectBivariateSpline(y_grid, x_grid, u)  # interpolador para u
        interp_v = RectBivariateSpline(y_grid, x_grid, v)  # interpolador para v

        # Actualizar posición de cada partícula
        for i in range(len(x_particles)):
            # Obtener velocidad interpolada en la posición de la partícula
            ux = interp_u(y_particles[i], x_particles[i])[0, 0]  # velocidad en x
            vy = interp_v(y_particles[i], x_particles[i])[0, 0]  # velocidad en y
            
            # Integrar posición usando método de Euler (x = x + v*dt)
            x_particles[i] += ux * dt * 2  # el *2 acelera la visualización
            y_particles[i] += vy * dt * 2
            
            # Condiciones de contorno periódicas y de reinicio
            if x_particles[i] > Lx:  # si sale por la derecha
                x_particles[i] = 0.0  # reaparece por la izquierda
                y_particles[i] = np.random.uniform(0.3, 0.7)  # nueva posición y aleatoria
            
            # Si sale por arriba o abajo, reinicia en posición aleatoria
            if y_particles[i] < 0 or y_particles[i] > Ly:
                y_particles[i] = np.random.uniform(0.3, 0.7)
            
            # Calcular color basado en la magnitud de la velocidad
            colors[i] = min(1.0, np.sqrt(ux**2 + vy**2) * 10)  # color según velocidad
        
        # Actualizar visualización de partículas
        scat.set_offsets(np.column_stack((x_particles, y_particles)))  # nuevas posiciones
        scat.set_array(colors)  # nuevos colores
        
        # Actualizar imagen de fondo con el campo de velocidad actual
        img.set_data(u)
        ax.set_title(f"Frame {frame+1}/{len(u_hist)}")
        
        return [img, scat]  # retornar elementos actualizados para blitting

    # Crear la animación
    ani = animation.FuncAnimation(fig, update, frames=len(u_hist), 
                                 interval=80, blit=True)
    
    # Guardar la animación como video
    ani.save("particulas_history.mp4", writer="ffmpeg", fps=30, dpi=150)
    plt.close(fig)  # cerrar figura para liberar memoria
    print("🎞️ Animación guardada como particulas_history.mp4")

if __name__ == "__main__":
    animate_from_history()