"""
# Vicsek model

Particles move at one speed and turn towards the mean heading of their neighbours, with a little noise. From random
headings they align into a flock that crosses the box together: the points take one color, which shows the heading.
"""

# %%
# Four hundred particles move `v0 = 0.5` per step in a periodic box of side 10, at density 4 and noise `eta = 0.1`.
# The blueprint shows the whole box and draws the particles larger than the model's visual does, so they read in the
# gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'vicsek'

box = rrb.Spatial2DView(
    origin='/flock',
    name='flock',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 10.0), y_range=(-10.0, 0.0)),
    overrides={'flock': rr.Points2D.from_fields(radii=0.15).visualizer()},
)
blueprint = rrb.Blueprint(
    box,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, n_particles=400, eta=0.1, seed=0)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0, observables=observables).run(until=150)

# %%
# Order against noise
# -------------------
# The Vicsek model orders at low noise and disorders at high noise. The stationary polar order parameter
# `|<exp(i theta)>|` should fall from near one at small `eta` towards the finite-size level `1 / sqrt(N)` as the noise
# grows, with the transition near `eta = 0.6` at this density.

noises = np.linspace(0.05, 0.95, 10)
orders = []
for noise in noises:
    model_system = create(MODEL, n_particles=400, eta=noise, seed=1)
    run = Simulation(model_system, dt=5.0, random_state=1, observables=describe(MODEL).observables_of(model_system))
    orders.append(np.mean(np.asarray(run.run(until=150).trajectory_['polarization'], dtype=float)[-10:]))
fig, (series, transition) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, np.asarray(simulation.trajectory_['polarization'], dtype=float))
series.set_ylim(0, 1)
series.set_xlabel('step')
series.set_ylabel('polar order')
series.set_title('Ordering at eta = 0.1')
transition.plot(noises, orders, 'o-', label='simulation')
transition.axhline(1 / np.sqrt(400), color='grey', linestyle=':', label='1 / sqrt(N)')
transition.set_ylim(0, 1)
transition.set_xlabel('noise eta')
transition.set_ylabel('stationary polar order')
transition.set_title('Order-disorder transition')
transition.legend()
fig.tight_layout()
plt.show()
