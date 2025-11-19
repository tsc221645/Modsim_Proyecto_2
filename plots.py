import matplotlib.pyplot as plt
import numpy as np
from physics import analytical_poiseuille

def plot_fields(u, v, p, Lx, Ly, fname="campos.png"):
    """
    Genera una figura con los campos de velocidad y presión para visualización.
    
    Args:
        u: Campo de velocidad en dirección x (2D array)
        v: Campo de velocidad en dirección y (2D array)
        p: Campo de presión (2D array)
        Lx, Ly: Dimensiones del dominio físico
        fname: Nombre del archivo para guardar la figura
    """
    # Crear figura con 2 subplots lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    
    # Primer subplot: Campo de velocidad u
    im0 = axs[0].imshow(u, 
                       origin="lower",        # Origen en esquina inferior izquierda
                       extent=[0, Lx, 0, Ly], # Escala física del dominio
                       aspect="auto",         # Ajustar aspecto automáticamente
                       cmap="turbo")          # Mapa de colores para mejor contraste
    axs[0].set_title("Velocidad u")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    fig.colorbar(im0, ax=axs[0], label="Velocidad")  # Barra de color para escala
    
    # Segundo subplot: Campo de presión
    im1 = axs[1].imshow(p, 
                       origin="lower",
                       extent=[0, Lx, 0, Ly],
                       aspect="auto")         # Mapa de colores por defecto (viridis)
    axs[1].set_title("Presión")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    fig.colorbar(im1, ax=axs[1], label="Presión")
    
    # Ajustar layout y guardar
    plt.tight_layout()  # Evita superposición de elementos
    plt.savefig(fname, dpi=150)  # Alta resolución para publicaciones
    plt.close()  # Cerrar figura para liberar memoria
    print(f"✅ Campos guardados en {fname}")


def plot_validation(u, Ly, nu, Fx, fname="validacion.png"):
    """
    Compara el perfil de velocidad numérico con la solución analítica de Poiseuille.
    
    Esta función valida que la simulación reproduce correctamente flujos con solución conocida.
    
    Args:
        u: Campo de velocidad completo (2D array)
        Ly: Altura del dominio
        nu: Viscosidad cinemática
        Fx: Fuerza externa por unidad de masa
        fname: Nombre del archivo para guardar la validación
    """
    ny = u.shape[0]  # Número de puntos en dirección y
    
    # Crear arrays para coordenadas y perfiles
    y = np.linspace(0, Ly, ny)  # Posiciones verticales
    
    # Extraer perfil vertical en el centro del dominio (x = Lx/2)
    u_col = u[:, u.shape[1] // 2]  # Velocidad en la columna central
    
    # Calcular solución analítica de referencia
    u_ref = analytical_poiseuille(y, Ly, nu, Fx)
    
    # Calcular error RMS (Root Mean Square) entre solución numérica y analítica
    err = np.sqrt(np.mean((u_col - u_ref) ** 2))
    
    # Crear figura de comparación
    plt.figure()
    
    # Graficar ambas soluciones
    plt.plot(u_col, y, label="Numérico", linewidth=2)          # Solución numérica
    plt.plot(u_ref, y, "--", label="Analítico", linewidth=2)   # Solución analítica (línea punteada)
    
    # Configurar gráfico
    plt.legend()
    plt.xlabel("Velocidad u")
    plt.ylabel("Posición y")
    plt.title(f"Validación Poiseuille (RMS = {err:.3e})")  # Mostrar error en título
    plt.grid(True, alpha=0.3)  # Grid sutil para mejor lectura
    
    # Guardar figura
    plt.savefig(fname, dpi=150)
    plt.close()
    
    print(f"Error RMS del perfil: {err:.4e}")

