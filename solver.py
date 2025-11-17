import numpy as np
from physics import build_rhs

# forcing_field (mantengo tu lógica pero con pequeñas garantías)
def forcing_field(params, t, ny, nx, dx, dy):
    Fx_field = np.zeros((ny, nx))
    Fy_field = np.zeros((ny, nx))
    sc = params.get("scenario", "poiseuille")

    if sc == "poiseuille":
        Fx_field[:] = params.get("Fx", 0.0)

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
        if params.get("jet_axis", "x") == "x":
            Fx_field = amp * G
        else:
            Fy_field = amp * G

    return Fx_field, Fy_field

def pressure_poisson(p, b, dx, dy, nit=1000, tol=1e-6):
    """Solución iterativa de Poisson (Jacobi) sobre interior con BC de Neumann (gradiente = 0)."""
    pn = p.copy()
    ny, nx = p.shape
    dx2 = dx*dx; dy2 = dy*dy
    denom = 2*(dx2 + dy2)

    for it in range(nit):
        pn[1:-1,1:-1] = (
            (pn[1:-1,2:] + pn[1:-1,:-2]) * dy2 +
            (pn[2:,1:-1] + pn[:-2,1:-1]) * dx2 -
            b[1:-1,1:-1] * dx2 * dy2
        ) / denom

        # Neumann BC: dp/dn = 0 -> set boundary equal to adjacent interior
        pn[:,0] = pn[:,1]
        pn[:,-1] = pn[:,-2]
        pn[0,:] = pn[1,:]
        pn[-1,:] = pn[-2,:]

        # residuo simple para criterio de corte (L_inf)
        res = np.max(np.abs(pn[1:-1,1:-1] - p[1:-1,1:-1]))
        p[:] = pn
        if res < tol:
            break
    # devuelve p y número de iteraciones (si quieres)
    return p

def step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx_field=None, Fy_field=None):
    """Avanza un paso de tiempo (explícito) con condiciones de contorno no deslizantes en top/bottom."""
    if Fx_field is None:
        Fx_field = np.zeros_like(u)
    if Fy_field is None:
        Fy_field = np.zeros_like(v)

    un = u.copy()
    vn = v.copy()

    # difusivo (segunda derivada) e convectivo (centrado)
    # interior indices
    i0, i1 = 1, -1
    j0, j1 = 1, -1

    # derivadas centradas en interior
    dudx = (un[1:-1,2:] - un[1:-1,:-2]) / (2*dx)
    dudy = (un[2:,1:-1] - un[:-2,1:-1]) / (2*dy)
    dvdx = (vn[1:-1,2:] - vn[1:-1,:-2]) / (2*dx)
    dvdy = (vn[2:,1:-1] - vn[:-2,1:-1]) / (2*dy)

    d2udx2 = (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]) / dx**2
    d2udy2 = (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]) / dy**2
    d2vdx2 = (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2]) / dx**2
    d2vdy2 = (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]) / dy**2

    dpdx = (p[1:-1,2:] - p[1:-1,:-2]) / (2*dx)
    dpdy = (p[2:,1:-1] - p[:-2,1:-1]) / (2*dy)

    u_new = un.copy()
    v_new = vn.copy()

    u_new[1:-1,1:-1] = (
        un[1:-1,1:-1] 
        - dt*(un[1:-1,1:-1]*dudx + vn[1:-1,1:-1]*dudy)
        - dt/rho * dpdx
        + nu*dt*(d2udx2 + d2udy2)
        + dt*Fx_field[1:-1,1:-1]
    )

    v_new[1:-1,1:-1] = (
        vn[1:-1,1:-1]
        - dt*(un[1:-1,1:-1]*dvdx + vn[1:-1,1:-1]*dvdy)
        - dt/rho * dpdy
        + nu*dt*(d2vdx2 + d2vdy2)
        + dt*Fy_field[1:-1,1:-1]
    )

    # Condiciones de frontera sencillas:
    # No-slip en paredes superior e inferior (y=0 y y=Ly)
    u_new[0,:] = 0.0
    u_new[-1,:] = 0.0
    v_new[0,:] = 0.0
    v_new[-1,:] = 0.0

    # Para left/right podemos usar Neumann (du/dx = 0) -> copiar valor interior
    u_new[:,0] = u_new[:,1]
    u_new[:,-1] = u_new[:,-2]
    v_new[:,0] = v_new[:,1]
    v_new[:,-1] = v_new[:,-2]

    return u_new, v_new

def apply_special_bcs(u, v, params):
    """Aplica condiciones especiales según escenario (mutates u,v)."""
    sc = params.get("scenario", "poiseuille")
    if sc == "lid_cavity":
        # top lid moving
        u[-1,:] = params.get("U_lid", 1.0)
        v[-1,:] = 0.0
        u[0,:] = 0.0
        v[0,:] = 0.0
    elif sc == "cylinder":
        nx, ny = params["nx"], params["ny"]
        x = np.linspace(0, params["Lx"], nx)
        y = np.linspace(0, params["Ly"], ny)
        X, Y = np.meshgrid(x, y)
        xc, yc = params["cyl_center"]
        r = params["cyl_radius"]
        mask = (X - xc) ** 2 + (Y - yc) ** 2 <= r ** 2
        u[mask] = 0.0
        v[mask] = 0.0
    # si hay más escenarios, agrégalos aquí


# ----- Simulación principal -----
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
