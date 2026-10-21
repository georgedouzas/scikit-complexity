"""
# Active Brownian particles

Self-propelled colloids swim at a constant speed along their own heading, which turns by rotational diffusion. Over
times shorter than the persistence time `1 / D_r` each particle moves in a nearly straight line; over longer times its
heading forgets its start and it diffuses, much faster than a passive particle. The view shows 200 swimmers spread
over a periodic square, each following its own persistent, wandering path.
"""

# %%
# The model is created by name from the catalog with speed `v = 3`, translational diffusion `D_t = 0.1`, and
# rotational diffusion `D_r = 1` in a square of side 50. The Euler-Maruyama stepper takes steps of 0.01 and the
# observer saves every frame of 0.1. The blueprint shows the square with the swimmers drawn larger than the model's
# visual does, so they read in the gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'active_brownian_particles'

system = create(MODEL, random_state=0)
view = rrb.Spatial2DView(
    origin='/swimmers',
    name='swimmers',
    visual_bounds=rrb.VisualBounds2D(x_range=(0, 50), y_range=(0, 50)),
    overrides={'swimmers': rr.Points2D.from_fields(radii=0.5).visualizer()},
)
blueprint = rrb.Blueprint(
    view,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
solver = EulerMaruyama(dt=0.01)
simulation = Simulation(system, solver=solver, dt=0.1, observers=[observer], random_state=0).run(until=20)

# %%
# From ballistic swimming to enhanced diffusion
# ---------------------------------------------
# The mean squared displacement of an active Brownian particle in the plane is
# `4 D_t t + (2 v^2 / D_r) (t - (1 - exp(-D_r t)) / D_r)`: ballistic, `v^2 t^2`, below `1 / D_r`, and diffusive with
# the effective coefficient `D_t + v^2 / (2 D_r)` above it. Particles in a box far larger than their excursions
# should follow this law for two rotational diffusion coefficients.

fig, ax = plt.subplots(figsize=(7, 4))
for rotation, color in ((1.0, 'tab:blue'), (0.1, 'tab:orange')):
    system_r = create(MODEL, n_particles=500, D_r=rotation, size=1e5, random_state=1)
    run = Simulation(system_r, solver=solver, dt=0.1, random_state=1).run(until=40)
    positions = run.trajectory_['swimmers.x']
    times = run.trajectory_.times[1:]
    squared = np.mean(np.sum((positions[1:] - positions[0]) ** 2, axis=-1), axis=1)
    exact = 4 * 0.1 * times + 2 * 9 / rotation * (times - (1 - np.exp(-rotation * times)) / rotation)
    ax.loglog(times, squared, 'o', markersize=3, color=color, label=f'simulation, D_r = {rotation}')
    ax.loglog(times, exact, color=color, label=f'exact, D_r = {rotation}')
ax.set_xlabel('time t')
ax.set_ylabel('mean squared displacement')
ax.set_title('Ballistic below 1 / D_r, diffusive above')
ax.legend()
fig.tight_layout()
plt.show()
