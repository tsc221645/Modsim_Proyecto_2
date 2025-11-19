import numpy as np

def continuity_equation(u, v, dx, dy):
    """
    Calcula el campo de divergencia ∇·u usando diferencias centrales en el interior.
    
    Esta función verifica la condición de incompresibilidad del flujo (∇·u = 0).
    
    Args:
        u: Campo de velocidad en dirección x (2D array)
        v: Campo de velocidad en dirección y (2D array) 
        dx: Espaciado de la malla en dirección x
        dy: Espaciado de la malla en dirección y
    
    Returns:
        div: Campo de divergencia (2D array)
    """
    div = np.zeros_like(u)
    
    # Cálculo en puntos interiores usando diferencias centrales
    # ∂u/∂x ≈ (u[i,j+1] - u[i,j-1]) / (2*dx)
    # ∂v/∂y ≈ (v[i+1,j] - v[i-1,j]) / (2*dy)
    div[1:-1,1:-1] = ((u[1:-1,2:] - u[1:-1,:-2])/(2*dx) +
                      (v[2:,1:-1] - v[:-2,1:-1])/(2*dy))
    
    # En los bordes se mantiene cero (condiciones de contorno)
    # Esto evita problemas con índices fuera de los límites
    return div

def build_rhs(u, v, dx, dy, dt, rho):
    """
    Construye el lado derecho b para la ecuación de Poisson de presión.
    
    Basado en la proyección del campo de velocidad para imponer incompresibilidad.
    La ecuación es: ∇²p = b, donde b se deriva de las ecuaciones de Navier-Stokes.
    
    Args:
        u, v: Campos de velocidad actuales
        dx, dy: Espaciado de la malla
        dt: Paso de tiempo
        rho: Densidad del fluido
    
    Returns:
        b: Término fuente para la ecuación de Poisson
    """
    ny, nx = u.shape
    b = np.zeros_like(u)

    # Cálculo de derivadas primeras usando diferencias centrales
    # Solo en puntos interiores para evitar condiciones de contorno
    dudx = (u[1:-1,2:] - u[1:-1,:-2]) / (2 * dx)  # ∂u/∂x
    dvdy = (v[2:,1:-1] - v[:-2,1:-1]) / (2 * dy)  # ∂v/∂y  
    dudy = (u[2:,1:-1] - u[:-2,1:-1]) / (2 * dy)  # ∂u/∂y
    dvdx = (v[1:-1,2:] - v[1:-1,:-2]) / (2 * dx)  # ∂v/∂x

    # Construcción del término fuente b según la formulación de proyección
    # b = ρ * [ (1/Δt)(∇·u) - ( (∂u/∂x)² + 2(∂u/∂y)(∂v/∂x) + (∂v/∂y)² ) ]
    b[1:-1,1:-1] = rho * ((1.0/dt) * (dudx + dvdy) -
                          (dudx**2 + 2*dudy*dvdx + dvdy**2))

    # Estabilización: fijar b en los bordes a cero
    # Esto corresponde a condiciones de Neumann homogéneas para la presión
    b[0,:] = 0.0    # Borde inferior
    b[-1,:] = 0.0   # Borde superior  
    b[:,0] = 0.0    # Borde izquierdo
    b[:,-1] = 0.0   # Borde derecho
    
    return b

def analytical_poiseuille(y, Ly, nu, Fx):
    """
    Calcula el perfil analítico parabólico de flujo de Poiseuille.
    
    Representa el flujo laminar completamente desarrollado entre dos placas paralelas
    con un gradiente de presión constante Fx.
    
    Args:
        y: Posición vertical (puede ser array)
        Ly: Altura del canal (distancia entre placas)
        nu: Viscosidad cinemática
        Fx: Fuerza externa por unidad de masa (gradiente de presión/ρ)
    
    Returns:
        Velocidad u(y) en cada posición y
    """
    # Solución analítica: u(y) = (Fx/(2ν)) * y * (Ly - y)
    # Perfil parabólico con máximo en el centro (y = Ly/2)
    return (Fx / (2.0 * nu)) * y * (Ly - y)