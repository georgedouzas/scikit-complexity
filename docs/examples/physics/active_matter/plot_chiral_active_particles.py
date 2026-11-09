"""
# Chiral active particles

Asymmetric swimmers, such as bacteria near a surface or L-shaped colloids, turn at a constant angular velocity
`omega` while they swim, so they trace circles of radius `v / omega`. Rotational noise slowly shifts the circles,
and over long times the particles diffuse, much more slowly than straight swimmers of the same speed. The view shows
200 swimmers circling in a periodic square.
"""

# %%
# The system is built from its class with speed `v = 3`, angular velocity `omega = 2`, translational
# diffusion `D_t = 0.1`, and rotational diffusion `D_r = 0.1` in a square of side 50. The Euler-Maruyama stepper takes
# steps of 0.01 and the observer saves every frame of 0.1. The blueprint shows the square with the swimmers drawn
# larger than the system's visual does, so they read in the gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import ChiralActiveParticlesSystem

SYSTEM = 'chiral_active_particles'

system = ChiralActiveParticlesSystem(random_state=0)
view = rrb.Spatial2DView(
    origin='/swimmers',
    name='swimmers',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0, 50), y_range=(-50, 0)),
    overrides={'swimmers': rr.Points2D.from_fields(radii=0.5).visualizer()},
)
blueprint = rrb.Blueprint(
    view,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
solver = EulerMaruyamaSolver(dt=0.01)
trajectory = simulate(system, solver=solver, dt=0.1, observers=[observer], random_state=0)

# %%
# Circling suppresses diffusion
# -----------------------------
# Van Teeffelen and Löwen derived the mean squared displacement of a chiral swimmer,
# `4 D_t t + 2 v^2 / w^2 [D_r w t + (omega^2 - D_r^2) + exp(-D_r t) ((D_r^2 - omega^2) cos(omega t)
# - 2 omega D_r sin(omega t))]` with `w = D_r^2 + omega^2`. It oscillates at the circling frequency before it turns
# diffusive with the coefficient `D_t + v^2 D_r / (2 w)`, far below that of straight swimmers. Particles in a box far
# larger than their circles should follow it for `omega = 0` and `omega = 2`.

fig, ax = plt.subplots(figsize=(7, 4))
rotation = 0.1
for omega, color in ((0.0, 'tab:blue'), (2.0, 'tab:orange')):
    system_o = ChiralActiveParticlesSystem(n_particles=500, omega=omega, size=1e5, random_state=1)
    run = simulate(system_o, until=40, solver=solver, dt=0.1, random_state=1)
    positions = run.get_values('swimmers.x')
    times = run.times[1:]
    squared = np.mean(np.sum((positions[1:] - positions[0]) ** 2, axis=-1), axis=1)
    w = rotation**2 + omega**2
    decay = np.exp(-rotation * times)
    bracket = rotation * w * times + omega**2 - rotation**2
    bracket += decay * ((rotation**2 - omega**2) * np.cos(omega * times) - 2 * omega * rotation * np.sin(omega * times))
    ax.loglog(times, squared, 'o', markersize=3, color=color, label=f'simulation, omega = {omega:.0f}')
    ax.loglog(times, 0.4 * times + 2 * 9 / w**2 * bracket, color=color, label=f'exact, omega = {omega:.0f}')
ax.set_xlabel('time t')
ax.set_ylabel('mean squared displacement')
ax.set_title('Chiral swimmers spread far less than straight ones')
ax.legend()
fig.tight_layout()
plt.show()
