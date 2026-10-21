"""
# Boids

The boids start scattered with random headings and gather into flocks that fly in one direction. Within a flock the
boids share a colour, which shows their horizontal velocity.
"""

# %%
# The boids fly on a square torus. The blueprint shows the whole torus and draws the boids larger than the model's
# visual does, so they read in the gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'boids'

flock = rrb.Spatial2DView(
    origin='/flock',
    name='flock',
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 40.0), y_range=(-40.0, 0.0)),
    overrides={'flock': rr.Points2D.from_fields(radii=0.55).visualizer()},
)
blueprint = rrb.Blueprint(
    flock,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, n_boids=150, size=40.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=200)

# %%
# Polarization
# ------------
# The polarization, the length of the mean heading of the boids, is near zero for random headings and one when all
# boids fly in the same direction. It should rise as alignment spreads through each flock. Separate flocks may fly
# in different directions, so it need not reach one.

polarization = []
for frame in range(len(simulation.trajectory_.times)):
    boids = simulation.trajectory_.agents('flock', frame)
    headings = np.column_stack([boids.values['vx'], boids.values['vy']])
    headings /= np.linalg.norm(headings, axis=1, keepdims=True)
    polarization.append(np.linalg.norm(headings.mean(axis=0)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(simulation.trajectory_.times, polarization)
ax.set_ylim(0, 1)
ax.set_xlabel('step')
ax.set_ylabel('polarization')
ax.set_title('Headings align')
plt.show()
