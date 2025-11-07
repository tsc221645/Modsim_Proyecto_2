import numpy as np

def default_params():
    """Devuelve un diccionario con parámetros del dominio y del fluido."""
    nx, ny = 121, 81
    Lx, Ly = 2.0, 1.0
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    params = {
        "nx": nx, "ny": ny, "Lx": Lx, "Ly": Ly,
        "dx": dx, "dy": dy,
        "rho": 1.0, "nu": 0.01,
        "dt": 0.0015, "nt": 5000, "nit": 60,
        "print_every": 200,

        # Escenario de flujo
        "scenario": "poiseuille",  # "poiseuille", "oscillatory_force", "jet", "lid_cavity", "cylinder"
        "Fx": 1.0,

        # Oscilatorio
        "F0": 1.0,
        "freq": 0.5,

        # Jet
        "jet_center": (0.2, 0.5),
        "jet_sigma": 0.05,
        "jet_strength": 50.0,
        "jet_axis": "x",

        # Lid cavity
        "U_lid": 1.0,

        # Cilindro
        "cyl_center": (1.0, 0.5),
        "cyl_radius": 0.1,
    }
    return params


def save_results(u, v, p, params, fname="resultados.npz"):
    """Guarda campos y parámetros."""
    np.savez(fname, u=u, v=v, p=p, **params)
    print(f"✅ Resultados guardados en {fname}")
