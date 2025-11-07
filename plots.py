import matplotlib.pyplot as plt
import numpy as np
from physics import analytical_poiseuille

def plot_fields(u, v, p, Lx, Ly, fname="campos.png"):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    im0 = axs[0].imshow(u, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto", cmap="turbo")
    axs[0].set_title("Velocidad u")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(p, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto")
    axs[1].set_title("Presión")
    fig.colorbar(im1, ax=axs[1])
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_validation(u, Ly, nu, Fx, fname="validacion.png"):
    ny = u.shape[0]
    y = np.linspace(0, Ly, ny)
    u_col = u[:, u.shape[1] // 2]
    u_ref = analytical_poiseuille(y, Ly, nu, Fx)
    err = np.sqrt(np.mean((u_col - u_ref) ** 2))

    plt.figure()
    plt.plot(u_col, y, label="Numérico")
    plt.plot(u_ref, y, "--", label="Analítico")
    plt.legend()
    plt.xlabel("u"); plt.ylabel("y")
    plt.title(f"Validación Poiseuille (RMS={err:.3e})")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Error RMS perfil: {err:.4e}")
