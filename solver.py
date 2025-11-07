import numpy as np
from physics import build_rhs

# ---------------------- Forzamientos ----------------------
def forcing_field(params, t, ny, nx, dx, dy):
    """Genera campos de fuerza Fx, Fy según el escenario."""
    Fx_field = np.zeros((ny, nx))
    Fy_field = np.zeros((ny, nx))
    sc = params.get("scenario", "poiseuille")

    if sc == "poiseuille":
        Fx_field[:] = params["Fx"]

    elif sc == "oscillatory_force":
        F0, f = params["F0"], params["freq"]
        Fx_field[:] = F0 * np.sin(2 * np.pi * f * t)

    elif sc == "jet":
        x = np.linspace(0, params["Lx"], nx)
        y = np.linspace(0, params["Ly"], ny)
        X, Y = np.meshgrid(x, y)
        x0, y0 = params["jet_center"]
        s = params["jet_sigma"]
        G = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s * s))
        amp = params["jet_strength"]
        if params["jet_axis"] == "x":
            Fx_field = amp * G
        else:
            Fy_field = amp * G

    return Fx_field, Fy_field


# ---------------------- Poisson Solver ----------------------
def pressure_poisson(p, b, dx, dy, nit):
    pn = np.empty_like(p)
    for _ in range(nit):
        pn[:] = p[:]
        p = (
            ((np.roll(pn, -1, axis=1) + np.roll(pn, 1, axis=1)) * dy**2
             + (np.roll(pn, -1, axis=0) + np.roll(pn, 1, axis=0)) * dx**2
             - b * dx**2 * dy**2)
            / (2 * (dx**2 + dy**2))
        )
        p[:, 0], p[:, -1] = p[:, -2], p[:, 1]
        p[0, :], p[-1, :] = p[1, :], p[-2, :]
    return p


# ---------------------- Paso de Navier-Stokes ----------------------
def step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx_field, Fy_field):
    un, vn = u.copy(), v.copy()

    dudx = (np.roll(un, -1, axis=1) - np.roll(un, 1, axis=1)) / (2 * dx)
    dudy = (np.roll(un, -1, axis=0) - np.roll(un, 1, axis=0)) / (2 * dy)
    dvdx = (np.roll(vn, -1, axis=1) - np.roll(vn, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(vn, -1, axis=0) - np.roll(vn, 1, axis=0)) / (2 * dy)

    d2udx2 = (np.roll(un, -1, axis=1) - 2 * un + np.roll(un, 1, axis=1)) / dx**2
    d2udy2 = (np.roll(un, -1, axis=0) - 2 * un + np.roll(un, 1, axis=0)) / dy**2
    d2vdx2 = (np.roll(vn, -1, axis=1) - 2 * vn + np.roll(vn, 1, axis=1)) / dx**2
    d2vdy2 = (np.roll(vn, -1, axis=0) - 2 * vn + np.roll(vn, 1, axis=0)) / dy**2

    dpdx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dx)
    dpdy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dy)

    u_new = (
        un - dt * (un * dudx + vn * dudy)
        - dt / rho * dpdx + nu * dt * (d2udx2 + d2udy2)
        + dt * Fx_field
    )
    v_new = (
        vn - dt * (un * dvdx + vn * dvdy)
        - dt / rho * dpdy + nu * dt * (d2vdx2 + d2vdy2)
        + dt * Fy_field
    )

    u_new[0, :], u_new[-1, :] = 0.0, 0.0
    v_new[0, :], v_new[-1, :] = 0.0, 0.0
    return u_new, v_new


# ---------------------- Condiciones especiales ----------------------
def apply_special_bcs(u, v, params, dx, dy):
    sc = params.get("scenario", "poiseuille")

    if sc == "lid_cavity":
        v[-1, :] = 0.0
        u[-1, :] = params["U_lid"]

    if sc == "cylinder":
        x = np.linspace(0, params["Lx"], params["nx"])
        y = np.linspace(0, params["Ly"], params["ny"])
        X, Y = np.meshgrid(x, y)
        xc, yc = params["cyl_center"]
        r = params["cyl_radius"]
        mask = (X - xc) ** 2 + (Y - yc) ** 2 <= r ** 2
        u[mask], v[mask] = 0.0, 0.0


# ---------------------- Simulación principal ----------------------
def simulate_flow(params):
    nx, ny = params["nx"], params["ny"]
    dx, dy = params["dx"], params["dy"]
    dt, nt, nit = params["dt"], params["nt"], params["nit"]
    rho, nu = params["rho"], params["nu"]

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    for n in range(1, nt + 1):
        t = n * dt
        b = build_rhs(u, v, dx, dy, dt, rho)
        p = pressure_poisson(p, b, dx, dy, nit)
        Fx_field, Fy_field = forcing_field(params, t, ny, nx, dx, dy)
        u, v = step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx_field, Fy_field)
        apply_special_bcs(u, v, params, dx, dy)

        if n % params["print_every"] == 0 or n == nt:
            div_x = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
            div_y = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
            div = np.mean(np.abs(div_x + div_y))
            print(f"Iter {n}/{nt} | div={div:.2e} | u_max={u.max():.3f}")

    return u, v, p
