from utils import default_params
from solver import simulate_flow
import numpy as np

if __name__ == "__main__":
    print("=== Simulación 2D Navier–Stokes (grabando estados) ===")
    params = default_params()
    params["nt"] = 5000   # número de pasos de tiempo
    params["print_every"] = 100
    params["scenario"] = "jet"   # flujo más dinámico

    nx, ny = params["nx"], params["ny"]
    u_hist, v_hist = [], []

    from physics import build_rhs
    from solver import pressure_poisson, step_navier_stokes, forcing_field

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    dx, dy, dt, rho, nu, nit = params["dx"], params["dy"], params["dt"], params["rho"], params["nu"], params["nit"]

    for n in range(1, params["nt"] + 1):
        b = build_rhs(u, v, dx, dy, dt, rho)
        p = pressure_poisson(p, b, dx, dy, nit)
        Fx, Fy = forcing_field(params, n * dt, ny, nx, dx, dy)
        u, v = step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx, Fy)
        if n % 20 == 0:
            u_hist.append(u.copy())
            v_hist.append(v.copy())
        if n % params["print_every"] == 0:
            print(f"Iter {n}/{params['nt']}")

    np.savez("historial_flujo.npz",
         u_hist=np.array(u_hist), v_hist=np.array(v_hist),
         Lx=params["Lx"], Ly=params["Ly"], dt=params["dt"],
         nx=params["nx"], ny=params["ny"])

    print("✅ Estados guardados en historial_flujo.npz")
