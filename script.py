from visualization import (
    animacion_u_mejorada,
    animacion_particulas_mejorada,
    animacion_streamlines,
    animacion_vorticidad,
    quiver_snapshot
)
# crear archivos MP4/PNG
animacion_u_mejorada(npz_path="historial_flujo.npz", fname="flujo_u_mejorado.mp4")
animacion_particulas_mejorada(npz_path="historial_flujo.npz", fname="particulas_mejorado.mp4")
animacion_streamlines(npz_path="historial_flujo.npz", fname="streamlines.mp4")
animacion_vorticidad(npz_path="historial_flujo.npz", fname="vorticidad.mp4")
quiver_snapshot(frame_index=-1, npz_path="historial_flujo.npz", fname="quiver_final.png")