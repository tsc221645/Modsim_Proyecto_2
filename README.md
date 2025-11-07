# Modelación y Simulación - Proyecto #2
## Simulación de dinámica de fluidos (Ecuaciones Navier Stokes en 2D)

### Integrantes
+ Sebastían García 22291
+ Ana Laura Tschen 221645

### Descripción
Este proyecto implementa la simulación numérica del flujo laminar incompresible bidimensional, resuelto mediante las ecuaciones de Navier–Stokes y la ecuación de continuidad.
El objetivo es observar el comportamiento dinámico del fluido bajo diferentes configuraciones de frontera y forzamiento, y visualizar el movimiento de partículas que siguen el campo de velocidades.

La simulación utiliza el método de diferencias finitas explícito (FTCS) con un esquema de presión tipo Poisson, siguiendo la metodología del curso CFD Python: 12 Steps to Navier–Stokes de Lorena Barba (2020).

### Estructura del Proyecto


### Fundamento Teórico

El fenómeno de la **dinámica de fluidos** se describe mediante las **ecuaciones de Navier–Stokes**, que provienen de aplicar las leyes de conservación de la **masa** y de la **cantidad de movimiento** a un volumen de fluido newtoniano.

Para un fluido **incompresible y viscoso**, las ecuaciones bidimensionales son:

$$
\frac{\partial u}{\partial t}
+ u \frac{\partial u}{\partial x}
+ v \frac{\partial u}{\partial y}
= -\frac{1}{\rho}\frac{\partial p}{\partial x}
+ \nu \left(
\frac{\partial^2 u}{\partial x^2}
+ \frac{\partial^2 u}{\partial y^2}
\right)
$$

$$
\frac{\partial v}{\partial t}
+ u \frac{\partial v}{\partial x}
+ v \frac{\partial v}{\partial y}
= -\frac{1}{\rho}\frac{\partial p}{\partial y}
+ \nu \left(
\frac{\partial^2 v}{\partial x^2}
+ \frac{\partial^2 v}{\partial y^2}
\right)
$$

donde:
- $(u(x,y,t))$ y $(v(x,y,t))$ son las componentes de la velocidad del fluido,  
- $(p(x,y,t))$ es la presión,  
- $(\rho)$ es la densidad constante del fluido,  
- $(\nu = \frac{\mu}{\rho})$ es la viscosidad cinemática.

Estas ecuaciones se complementan con la **ecuación de continuidad**, que asegura la conservación de la masa:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

La ecuación de Navier–Stokes expresa el equilibrio entre las fuerzas de **inercia**, **presión**, **viscosas** y **externas** que actúan sobre el fluido:

$$
\rho \left( \frac{\partial \vec{u}}{\partial t}
+ (\vec{u}\cdot\nabla)\vec{u} \right)
= -\nabla p + \mu \nabla^2 \vec{u} + \vec{f}
$$

donde \(\vec{f}\) representa las fuerzas externas aplicadas (por ejemplo, un gradiente de presión o una aceleración).



### Ecuación de Poisson para la presión

En la formulación numérica usada, la presión se obtiene resolviendo una ecuación de Poisson derivada de la continuidad:

$$
\nabla^2 p = \rho
\left[
\frac{1}{\Delta t}
\left(
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}
\right)
- \left(\frac{\partial u}{\partial x}\right)^2
- 2 \frac{\partial u}{\partial y}\frac{\partial v}{\partial x}
- \left(\frac{\partial v}{\partial y}\right)^2
\right]
$$


### Método numérico

Las derivadas espaciales y temporales se aproximan mediante el **método de diferencias finitas** con un esquema explícito **FTCS (Forward Time, Central Space)**:

$$
\frac{\partial \phi}{\partial t}
\approx \frac{\phi^{n+1}_{i,j} - \phi^n_{i,j}}{\Delta t},
\quad
\frac{\partial \phi}{\partial x}
\approx \frac{\phi^n_{i+1,j} - \phi^n_{i-1,j}}{2\Delta x}
$$

De esta forma, el modelo simula la evolución del campo de velocidades \((u,v)\) en el tiempo para distintos escenarios de flujo: **Poiseuille**, **flujo oscilatorio**, o **chorro localizado (jet)**.


### Escenarios simulados
+ poiseuille	- Flujo laminar clásico entre placas paralelas.
+ oscillatory_force	- Flujo impulsado por una fuerza sinusoidal en el tiempo.
+ jet	- Chorro localizado que impulsa el fluido desde una región específica.
+ lid_cavity -	Cavidad con tapa móvil (vórtices inducidos).
+ cylinder -	Flujo con obstáculo cilíndrico fijo.

### Dependencias
Para poder correr el proyecto, se necesitan las siguientes dependencias:
+ numpy
+ matplotlib
+ scipy
+ tqdm
+ ffmpeg-python
Estas se pueden instalar corriendo ``` pip install -r requirements.txt ```


### Ejecución
* Para simular el flujo (campo u,v,p) - ```python main.py```. Esto genera el archivo ``` resultados.npz``` que contiene los campos finales, la imagen ``` campos.png``` como visualización y la imagen ```validacion.png``` que muestra una comparación numérica vs analítica (Poiseuille).
* Para simular la animación del flujo - ```python animation.py```. Esto generará un video ```flujo_animacion.mp4``` que muestra la evolución del campo de velocidad
* Para visualizar la animación con partículas dinámicas (flujo + partículas) - ```python particles.py```. Mostrará las partículas siguiendo el flujo en tiempo real y generará un video ```particulas_flujo.mp4```.
* Para ver la animación rápida desde los resultados guardados - ``` python particles_from_history.py```. En esta animación se utiliza el historial del flujo y se produce el video ```particulas_history.mp4```

### Referencias

Barba, L. A. (2020). CFD Python: 12 Steps to Navier–Stokes. The George Washington University.
https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes

White, F. M. (2011). Fluid Mechanics (7th ed.). McGraw-Hill.

Fox, R. W., Pritchard, P. J., & McDonald, A. T. (2015). Introduction to Fluid Mechanics (9th ed.). Wiley.