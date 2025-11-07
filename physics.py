import numpy as np

def continuity_equation(u, v, dx, dy):
    """Devuelve el campo de divergencia ∇·u."""
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
    return dudx + dvdy


def build_rhs(u, v, dx, dy, dt, rho):
    """Construye el lado derecho de la ecuación de Poisson."""
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)

    b = rho * ((1.0 / dt) * (dudx + dvdy) - (dudx ** 2 + 2 * dudy * dvdx + dvdy ** 2))
    b[0, :], b[-1, :] = 0.0, 0.0
    return b


def analytical_poiseuille(y, Ly, nu, Fx):
    """Perfil analítico parabólico de Poiseuille."""
    return (Fx / (2.0 * nu)) * y * (Ly - y)
