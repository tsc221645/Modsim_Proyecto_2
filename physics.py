import numpy as np

def continuity_equation(u, v, dx, dy):
    """Devuelve el campo de divergencia ∇·u usando diferencias centrales en interior."""
    div = np.zeros_like(u)
    # interior
    div[1:-1,1:-1] = ((u[1:-1,2:] - u[1:-1,:-2])/(2*dx) +
                      (v[2:,1:-1] - v[:-2,1:-1])/(2*dy))
    # boundaries: keep zeros or one-sided if needed (here we leave zeros)
    return div

def build_rhs(u, v, dx, dy, dt, rho):
    """Construye el lado derecho b para la ecuación de Poisson (discretización centrada)."""
    ny, nx = u.shape
    b = np.zeros_like(u)

    # derivadas en interior (centradas)
    dudx = (u[1:-1,2:] - u[1:-1,:-2]) / (2 * dx)
    dvdy = (v[2:,1:-1] - v[:-2,1:-1]) / (2 * dy)
    dudy = (u[2:,1:-1] - u[:-2,1:-1]) / (2 * dy)
    dvdx = (v[1:-1,2:] - v[1:-1,:-2]) / (2 * dx)

    b[1:-1,1:-1] = rho * ((1.0/dt) * (dudx + dvdy) -
                          (dudx**2 + 2*dudy*dvdx + dvdy**2))

    # for stability, set boundary b to zero (alternative: one-sided derivatives)
    b[0,:] = 0.0
    b[-1,:] = 0.0
    b[:,0] = 0.0
    b[:,-1] = 0.0
    return b

def analytical_poiseuille(y, Ly, nu, Fx):
    """Perfil analítico parabólico de Poiseuille (u(y))."""
    return (Fx / (2.0 * nu)) * y * (Ly - y)
