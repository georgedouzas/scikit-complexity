"""
# Grégoire-Chaté cohesive flocks

Particles that align with their neighbours and also keep a preferred distance from them form a cohesive flock that
travels as one: the points take one color, the common heading, at nearly even spacings.
"""

# %%
# Two hundred particles move `v0 = 0.05` per step in a periodic box of side 5, with alignment `alpha = 1`, cohesion
# `beta = 0.5`, and vectorial noise `eta = 0.1`. The blueprint shows the whole box and draws the particles larger than
# the model's visual does, so they read in the gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'gregoire_chate'

box = rrb.Spatial2DView(
    origin='/flock',
    name='flock',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 5.0), y_range=(-5.0, 0.0)),
    overrides={'flock': rr.Points2D.from_fields(radii=0.08).visualizer()},
)
blueprint = rrb.Blueprint(
    box,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, n_particles=200, seed=0)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0, observables=observables).run(until=300)

# %%
# Spacing inside the flock
# ------------------------
# The cohesion force repels particles closer than `r_c = 0.2` and pulls those beyond the preferred distance
# `r_e = 0.5` back up to `r_a = 0.8`. The distances of the particles to their nearest neighbour should therefore settle
# above `r_c`, with no pair closer than the repulsive core, while the polar order grows towards one.

positions = np.asarray(simulation.trajectory_['flock.x'][-1])
difference = positions[:, None, :] - positions[None, :, :]
difference -= 5.0 * np.rint(difference / 5.0)
distance = np.sqrt((difference**2).sum(axis=-1))
np.fill_diagonal(distance, np.inf)
fig, (series, spacing) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, np.asarray(simulation.trajectory_['polarization'], dtype=float))
series.set_ylim(0, 1)
series.set_xlabel('step')
series.set_ylabel('polar order')
series.set_title('Ordering of the cohesive flock')
spacing.hist(distance.min(axis=1), bins=30)
spacing.axvline(0.2, color='black', linestyle='--', label='r_c')
spacing.axvline(0.5, color='grey', linestyle=':', label='r_e')
spacing.set_xlabel('distance to the nearest neighbour')
spacing.set_ylabel('particles')
spacing.set_title('Spacing at the last step')
spacing.legend()
fig.tight_layout()
plt.show()
