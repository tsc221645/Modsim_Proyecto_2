import numpy as np
from physics import build_rhs

def forcing_field(params, t, ny, nx, dx, dy):
    """
    Calcula campos de fuerza externa para diferentes escenarios de flujo.
    
    Args:
        params: Diccionario de parámetros de simulación
        t: Tiempo actual (para fuerzas dependientes del tiempo)
        ny, nx: Dimensiones de la malla
        dx, dy: Espaciado de la malla
    
    Returns:
        Fx_field, Fy_field: Campos de fuerza en direcciones x e y
    """
    Fx_field = np.zeros((ny, nx))
    Fy_field = np.zeros((ny, nx))
    sc = params.get("scenario", "poiseuille")  # Escenario por defecto

    if sc == "poiseuille":
        # Fuerza constante para flujo de Poiseuille (canal)
        Fx_field[:] = params.get("Fx", 0.0)

    elif sc == "oscillatory_force":
        # Fuerza oscilatoria en el tiempo (flujos pulsantes)
        F0, f = params["F0"], params["freq"]
        Fx_field[:] = F0 * np.sin(2 * np.pi * f * t)

    elif sc == "jet":
        # Chorro localizado (Gaussiano)
        x = np.linspace(0, params["Lx"], nx)
        y = np.linspace(0, params["Ly"], ny)
        X, Y = np.meshgrid(x, y)
        x0, y0 = params["jet_center"]  # Centro del chorro
        s = params["jet_sigma"]        # Ancho del chorro
        # Perfil Gaussiano del chorro
        G = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s * s))
        amp = params["jet_strength"]   # Intensidad del chorro
        if params.get("jet_axis", "x") == "x":
            Fx_field = amp * G  # Chorros en dirección x
        else:
            Fy_field = amp * G  # Chorros en dirección y

    return Fx_field, Fy_field

def pressure_poisson(p, b, dx, dy, nit=1000, tol=1e-6):
    """
    Resuelve la ecuación de Poisson para la presión usando el método iterativo de Jacobi.
    
    Ecuación: ∇²p = b con condiciones de Neumann (dp/dn = 0) en los bordes.
    
    Args:
        p: Campo de presión inicial (se actualiza in-place)
        b: Término fuente de la ecuación de Poisson
        dx, dy: Espaciado de la malla
        nit: Número máximo de iteraciones
        tol: Tolerancia para convergencia
    
    Returns:
        p: Campo de presión resuelto
    """
    pn = p.copy()  # Copia para iteración
    ny, nx = p.shape
    dx2 = dx*dx; dy2 = dy*dy
    denom = 2*(dx2 + dy2)  # Denominador común

    for it in range(nit):
        # Método de Jacobi: p_new = (p_E + p_W)*dy² + (p_N + p_S)*dx² - b*dx²*dy² / denominador
        pn[1:-1,1:-1] = (
            (pn[1:-1,2:] + pn[1:-1,:-2]) * dy2 +      # Términos este-oeste
            (pn[2:,1:-1] + pn[:-2,1:-1]) * dx2 -      # Términos norte-sur
            b[1:-1,1:-1] * dx2 * dy2                  # Término fuente
        ) / denom

        # Condiciones de Neumann: gradiente normal cero en bordes
        # Esto hace que la presión esté definida hasta una constante
        pn[:,0] = pn[:,1]      # Borde izquierdo
        pn[:,-1] = pn[:,-2]    # Borde derecho
        pn[0,:] = pn[1,:]      # Borde inferior
        pn[-1,:] = pn[-2,:]    # Borde superior

        # Criterio de convergencia: norma infinito del residuo
        res = np.max(np.abs(pn[1:-1,1:-1] - p[1:-1,1:-1]))
        p[:] = pn  # Actualizar presión
        if res < tol:
            break
    return p

def step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx_field=None, Fy_field=None):
    """
    Avanza las ecuaciones de Navier-Stokes un paso en el tiempo usando esquema explícito.
    
    Forma discreta: u^{n+1} = u^n - Δt(u·∇)u - (Δt/ρ)∇p + νΔt∇²u + Δt·F
    
    Args:
        u, v: Campos de velocidad actuales
        p: Campo de presión
        dx, dy, dt: Parámetros discretos
        rho, nu: Parámetros físicos (densidad, viscosidad)
        Fx_field, Fy_field: Campos de fuerza externa
    
    Returns:
        u_new, v_new: Campos de velocidad actualizados
    """
    if Fx_field is None:
        Fx_field = np.zeros_like(u)
    if Fy_field is None:
        Fy_field = np.zeros_like(v)

    un = u.copy()  # Velocidad en tiempo n
    vn = v.copy()

    # Cálculo de derivadas usando diferencias finitas centradas
    # Solo en puntos interiores para evitar condiciones de contorno
    dudx = (un[1:-1,2:] - un[1:-1,:-2]) / (2*dx)  # ∂u/∂x
    dudy = (un[2:,1:-1] - un[:-2,1:-1]) / (2*dy)  # ∂u/∂y
    dvdx = (vn[1:-1,2:] - vn[1:-1,:-2]) / (2*dx)  # ∂v/∂x
    dvdy = (vn[2:,1:-1] - vn[:-2,1:-1]) / (2*dy)  # ∂v/∂y

    # Términos difusivos (Laplaciano)
    d2udx2 = (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]) / dx**2
    d2udy2 = (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]) / dy**2
    d2vdx2 = (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2]) / dx**2
    d2vdy2 = (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]) / dy**2

    # Gradiente de presión
    dpdx = (p[1:-1,2:] - p[1:-1,:-2]) / (2*dx)
    dpdy = (p[2:,1:-1] - p[:-2,1:-1]) / (2*dy)

    u_new = un.copy()
    v_new = vn.copy()

    # Actualización de velocidad u (componente x)
    u_new[1:-1,1:-1] = (
        un[1:-1,1:-1] 
        - dt*(un[1:-1,1:-1]*dudx + vn[1:-1,1:-1]*dudy)  # Término convectivo (u·∇)u
        - dt/rho * dpdx                                  # Gradiente de presión
        + nu*dt*(d2udx2 + d2udy2)                       # Término difusivo
        + dt*Fx_field[1:-1,1:-1]                        # Fuerza externa
    )

    # Actualización de velocidad v (componente y)
    v_new[1:-1,1:-1] = (
        vn[1:-1,1:-1]
        - dt*(un[1:-1,1:-1]*dvdx + vn[1:-1,1:-1]*dvdy)  # Término convectivo
        - dt/rho * dpdy                                  # Gradiente de presión
        + nu*dt*(d2vdx2 + d2vdy2)                       # Término difusivo
        + dt*Fy_field[1:-1,1:-1]                        # Fuerza externa
    )

    # Condiciones de contorno: no-deslizamiento en paredes superior e inferior
    u_new[0,:] = 0.0   # Pared inferior
    u_new[-1,:] = 0.0  # Pared superior
    v_new[0,:] = 0.0
    v_new[-1,:] = 0.0

    # Condiciones de salida (Neumann) en bordes izquierdo/derecho
    u_new[:,0] = u_new[:,1]      # Gradiente cero en borde izquierdo
    u_new[:,-1] = u_new[:,-2]    # Gradiente cero en borde derecho
    v_new[:,0] = v_new[:,1]
    v_new[:,-1] = v_new[:,-2]

    return u_new, v_new

def apply_special_bcs(u, v, params, dx, dy):
    """
    Aplica condiciones de contorno especiales para diferentes escenarios.
    
    Args:
        u, v: Campos de velocidad (se modifican in-place)
        params: Parámetros que definen el escenario
    """
    sc = params.get("scenario", "poiseuille")
    if sc == "lid_cavity":
        # Cavidad con tapa móvil: tapa superior se mueve con velocidad U_lid
        u[-1,:] = params.get("U_lid", 1.0)  # Tapa móvil
        v[-1,:] = 0.0                       # Sin flujo vertical en tapa
        u[0,:] = 0.0                        # Fondo estacionario
        v[0,:] = 0.0
    elif sc == "cylinder":
        # Flujo alrededor de cilindro: velocidad cero dentro del cilindro
        nx, ny = params["nx"], params["ny"]
        x = np.linspace(0, params["Lx"], nx)
        y = np.linspace(0, params["Ly"], ny)
        X, Y = np.meshgrid(x, y)
        xc, yc = params["cyl_center"]  # Centro del cilindro
        r = params["cyl_radius"]       # Radio del cilindro
        # Crear máscara para puntos dentro del cilindro
        mask = (X - xc) ** 2 + (Y - yc) ** 2 <= r ** 2
        u[mask] = 0.0  # Condición de no-deslizamiento en cilindro
        v[mask] = 0.0

def simulate_flow(params):
    """
    Función principal que ejecuta la simulación completa de Navier-Stokes.
    
    Implementa el método de proyección (fractional step):
    1. Calcular término derecho para Poisson
    2. Resolver para presión
    3. Avanzar velocidad
    4. Aplicar condiciones de contorno
    
    Args:
        params: Diccionario con todos los parámetros de simulación
    
    Returns:
        u, v, p: Campos finales de velocidad y presión
    """
    nx, ny = params["nx"], params["ny"]
    dx, dy = params["dx"], params["dy"]
    dt, nt, nit = params["dt"], params["nt"], params["nit"]
    rho, nu = params["rho"], params["nu"]

    # Inicializar campos
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Bucle temporal principal
    for n in range(1, nt + 1):
        t = n * dt
        
        # Paso 1: Calcular término fuente para ecuación de Poisson
        b = build_rhs(u, v, dx, dy, dt, rho)
        
        # Paso 2: Resolver ecuación de Poisson para presión
        p = pressure_poisson(p, b, dx, dy, nit)
        
        # Paso 3: Calcular fuerzas externas
        Fx_field, Fy_field = forcing_field(params, t, ny, nx, dx, dy)
        
        # Paso 4: Avanzar ecuaciones de Navier-Stokes
        u, v = step_navier_stokes(u, v, p, dx, dy, dt, rho, nu, Fx_field, Fy_field)
        
        # Paso 5: Aplicar condiciones de contorno especiales
        apply_special_bcs(u, v, params, dx, dy)

        # Monitoreo del progreso y verificación de incompresibilidad
        if n % params["print_every"] == 0 or n == nt:
            # Calcular divergencia para verificar incompresibilidad
            div_x = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
            div_y = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
            div = np.mean(np.abs(div_x + div_y))  # Promedio de |∇·u|
            print(f"Iter {n}/{nt} | div={div:.2e} | u_max={u.max():.3f}")

    return u, v, p