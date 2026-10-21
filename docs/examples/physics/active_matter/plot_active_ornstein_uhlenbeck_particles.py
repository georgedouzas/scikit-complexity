"""
# Active Ornstein-Uhlenbeck particles

The propulsion of each particle is an Ornstein-Uhlenbeck process, a random velocity that keeps its direction and size
for the persistence time `tau` before it changes. The particles therefore move in smooth, persistent excursions over
short times and diffuse with the coefficient `D` over long times, a simple model of active colloids and cells. The
view shows 200 particles spread over a periodic square.
"""

# %%
# The model is created by name from the catalog with persistence time `tau = 1`, active diffusion `D = 1`, and no
# thermal diffusion, in a square of side 50. The Euler-Maruyama stepper takes steps of 0.01 and the observer saves
# every frame of 0.1. The blueprint shows the square with the particles drawn larger than the model's visual does, so
# they read in the gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'active_ornstein_uhlenbeck_particles'

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
# Persistent motion and long-time diffusion
# -----------------------------------------
# The velocity correlation `<v(0) v(t)> = (D / tau) exp(-|t| / tau)` per axis gives the mean squared displacement
# in the plane `4 D (t - tau (1 - exp(-t / tau)))`: ballistic, `2 D t^2 / tau`, below `tau`, and diffusive, `4 D t`,
# above it. Particles in a box far larger than their excursions should follow this law for two persistence times.

fig, ax = plt.subplots(figsize=(7, 4))
for tau, color in ((1.0, 'tab:blue'), (10.0, 'tab:orange')):
    system_t = create(MODEL, n_particles=500, tau=tau, size=1e5, random_state=1)
    run = Simulation(system_t, solver=solver, dt=0.1, random_state=1).run(until=40)
    positions = run.trajectory_['swimmers.x']
    times = run.trajectory_.times[1:]
    squared = np.mean(np.sum((positions[1:] - positions[0]) ** 2, axis=-1), axis=1)
    exact = 4 * (times - tau * (1 - np.exp(-times / tau)))
    ax.loglog(times, squared, 'o', markersize=3, color=color, label=f'simulation, tau = {tau:.0f}')
    ax.loglog(times, exact, color=color, label=f'exact, tau = {tau:.0f}')
ax.set_xlabel('time t')
ax.set_ylabel('mean squared displacement')
ax.set_title('Ballistic below tau, diffusive above')
ax.legend()
fig.tight_layout()
plt.show()
