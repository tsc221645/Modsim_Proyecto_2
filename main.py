from utils import default_params
# Importar funciones específicas para física y resolución
from physics import build_rhs
from solver import pressure_poisson, step_navier_stokes, forcing_field
import numpy as np

if __name__ == "__main__":
    print("=== Simulación 2D Navier–Stokes (grabando estados) ===")
    
    # Obtener parámetros por defecto para la simulación
    params = default_params()
    
    # Configurar parámetros específicos para esta simulación
    params["nt"] = 5000   # número de pasos de tiempo totales
    params["print_every"] = 100  # frecuencia para imprimir progreso
    params["scenario"] = "jet"   # tipo de escenario (flujo más dinámico)

    # Obtener dimensiones de la malla
    nx, ny = params["nx"], params["ny"]
    
    # Listas para almacenar el historial de velocidades
    u_hist, v_hist = [], []

    # Inicializar campos de velocidad (u, v) y presión (p) a cero
    u = np.zeros((ny, nx))  # componente x de la velocidad
    v = np.zeros((ny, nx))  # componente y de la velocidad  
    p = np.zeros((ny, nx))  # campo de presión

    # Extraer parámetros físicos y numéricos
    dx, dy, dt, rho, nu, nit = params["dx"], params["dy"], params["dt"], params["rho"], params["nu"], params["nit"]

    # Bucle principal de la simulación - avanza en el tiempo
    for n in range(1, params["nt"] + 1):
        # Paso 1: Construir el término derecho para la ecuación de Poisson
        b = build_rhs(u, v, dx, dy, dt, rho)
        
        # Paso 2: Resolver la ecuación de Poisson para la presión
        p = pressure_poisson(p, b, dx, dy, nit)
        
        # Paso 3: Calcular campo de fuerzas externas (si las hay)
        Fx, Fy = forcing_field(params, n * dt, ny, nx, dx, dy)
        
        # Paso 4: Avanzar las ecuaciones de Navier-Stokes un paso en el tiempo
        u, v = step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx, Fy)
        
        # Guardar estado cada 20 pasos de tiempo para análisis/post-procesamiento
        if n % 20 == 0:
            u_hist.append(u.copy())  # usar copy() para evitar referencias
            v_hist.append(v.copy())
            
        # Mostrar progreso cada ciertos pasos
        if n % params["print_every"] == 0:
            print(f"Iter {n}/{params['nt']}")

    # Guardar todos los datos de la simulación en un archivo
    np.savez("historial_flujo.npz",
         u_hist=np.array(u_hist), v_hist=np.array(v_hist),  # convertir listas a arrays
         Lx=params["Lx"], Ly=params["Ly"], dt=params["dt"],  # parámetros geométricos
         nx=params["nx"], ny=params["ny"])  # resolución de la malla

    print("✅ Estados guardados en historial_flujo.npz")