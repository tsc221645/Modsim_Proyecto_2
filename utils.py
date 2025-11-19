import numpy as np

def default_params():
    """
    Devuelve un diccionario con parámetros por defecto para la simulación de Navier-Stokes.
    
    Este diccionario centraliza toda la configuración de la simulación, incluyendo:
    - Parámetros del dominio computacional
    - Propiedades del fluido
    - Configuración temporal
    - Escenarios de flujo específicos
    
    Returns:
        params: Diccionario con todos los parámetros de simulación
    """
    # Resolución de la malla (número de puntos en x e y)
    nx, ny = 121, 81
    
    # Dimensiones físicas del dominio [metros]
    Lx, Ly = 2.0, 1.0  # Dominio rectangular 2:1
    
    # Espaciamiento de la malla (diferenciales espaciales)
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # -1 porque hay nx puntos, nx-1 intervalos

    # Diccionario principal de parámetros
    params = {
        # --- Geometría y malla ---
        "nx": nx, "ny": ny,      # Puntos de la malla en x e y
        "Lx": Lx, "Ly": Ly,      # Dimensiones físicas del dominio
        "dx": dx, "dy": dy,      # Diferenciales espaciales
        
        # --- Propiedades del fluido ---
        "rho": 1.0,              # Densidad [kg/m³] (normalizada)
        "nu": 0.01,              # Viscosidad cinemática [m²/s]
        
        # --- Parámetros temporales ---
        "dt": 0.0015,            # Paso de tiempo [s] - debe cumplir CFL
        "nt": 5000,              # Número total de pasos de tiempo
        "nit": 200,              # Iteraciones máximas para Poisson
        
        # --- Configuración de salida ---
        "print_every": 200,      # Frecuencia de impresión en consola
        
        # --- ESCENARIOS DE FLUJO ---
        # "poiseuille", "oscillatory_force", "jet", "lid_cavity", "cylinder"
        "scenario": "poiseuille",
        
        # === PARÁMETROS ESPECÍFICOS POR ESCENARIO ===
        
        # --- Flujo de Poiseuille (canal) ---
        "Fx": 1.0,               # Fuerza externa por unidad de masa [m/s²]
                                 # Conduce el flujo en dirección x
        
        # --- Fuerza oscilatoria ---
        "F0": 1.0,               # Amplitud de la fuerza oscilatoria [m/s²]
        "freq": 0.5,             # Frecuencia de oscilación [Hz]
        
        # --- Configuración de chorro (jet) ---
        "jet_center": (0.2, 0.5),  # Centro del chorro (x, y) [m]
        "jet_sigma": 0.05,       # Ancho del chorro Gaussiano [m]
        "jet_strength": 50.0,    # Intensidad del chorro [m/s²]
        "jet_axis": "x",         # Dirección del chorro ("x" o "y")
        
        # --- Cavidad con tapa móvil (lid-driven cavity) ---
        "U_lid": 1.0,            # Velocidad de la tapa superior [m/s]
        
        # --- Flujo alrededor de cilindro ---
        "cyl_center": (1.0, 0.5),  # Centro del cilindro (x, y) [m]
        "cyl_radius": 0.1,       # Radio del cilindro [m]
    }
    return params

def compute_stable_dt(u, v, dx, dy, nu, cfl=0.4):
    """
    Calcula un paso de tiempo estable basado en restricciones CFL y viscosidad.
    
    Para que la simulación sea estable, el paso de tiempo debe satisfacer:
    1. Condición CFL: dt ≤ CFL * min(dx/|u|_max, dy/|v|_max)
    2. Condición viscosa: dt ≤ 0.25 * min(dx², dy²) / ν
    
    Args:
        u, v: Campos de velocidad actuales [m/s]
        dx, dy: Espaciamiento de la malla [m]
        nu: Viscosidad cinemática [m²/s]
        cfl: Número de Courant-Friedrichs-Lewy (típicamente 0.2-0.5)
    
    Returns:
        dt: Paso de tiempo estable calculado [s]
    """
    # Velocidades máximas en cada dirección (con valor mínimo para evitar división por cero)
    umax = max(1e-8, np.max(np.abs(u)))  # Mínimo 1e-8 para casos de flujo en reposo
    vmax = max(1e-8, np.max(np.abs(v)))
    
    # Restricción por convección (CFL condition)
    # Asegura que una partícula no viaje más de una celda por paso de tiempo
    dt_conv = cfl * min(dx / umax, dy / vmax)
    
    # Restricción por difusión viscosa
    # Asegura estabilidad del esquema explícito para términos difusivos
    dt_diff = 0.25 * min(dx*dx, dy*dy) / nu
    
    # El paso de tiempo final es el más restrictivo de ambas condiciones
    return min(dt_conv, dt_diff)