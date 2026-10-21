"""
# Diffusion in a speckle field

Colloids diffuse through the random light pattern of a speckle field, whose bright hotspots pull them in by optical
gradient forces. In a strong field a particle is caught by a hotspot, rattles there for a long time, and only
occasionally escapes to the next one, so its diffusion becomes anomalous and slower. The view shows 300 particles
gathered in clumps at the hotspots of the field.
"""

# %%
# The model is created by name from the catalog with 50 hotspots in a square of side 100 and the field strength
# 30. The Euler-Maruyama stepper takes steps of 0.01 and the observer saves every frame of 0.25. The blueprint shows
# the particles over the square of the hotspots, drawn larger than the model's visual does, so they read in the
# gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'diffusion_in_a_speckle_field'

system = create(MODEL, n_particles=300, strength=30.0, random_state=0)
view = rrb.Spatial2DView(
    origin='/beads',
    name='beads',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0, 100), y_range=(-100, 0)),
    overrides={'beads': rr.Points2D.from_fields(radii=0.8).visualizer()},
)
blueprint = rrb.Blueprint(
    view,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
solver = EulerMaruyama(dt=0.01)
simulation = Simulation(system, solver=solver, dt=0.25, observers=[observer], random_state=0).run(until=60)

# %%
# Trapping slows diffusion
# ------------------------
# Without a field the mean squared displacement in the plane is `4 D t`. Measured from time 10, once the particles
# have fallen into the hotspots, the mean squared displacement should fall further below the free line as the field
# grows stronger and the hotspots hold the particles for longer: a plateau broken by rare escapes in the field of
# strength 30, and a flat plateau, the rattling inside one hotspot, in the field of strength 100.

fig, ax = plt.subplots(figsize=(7, 4))
for strength in (0.0, 30.0, 100.0):
    run = Simulation(create(MODEL, n_particles=300, strength=strength), solver=solver, dt=0.25, random_state=1)
    positions = run.run(until=70).trajectory_['beads.x'][40:]
    times = run.trajectory_.times[41:] - 10
    squared = np.mean(np.sum((positions[1:] - positions[0]) ** 2, axis=-1), axis=1)
    ax.loglog(times, squared, label=f'strength {strength:.0f}')
ax.loglog(times, 4 * times, '--', color='black', label='free diffusion, 4 D t')
ax.set_xlabel('time t')
ax.set_ylabel('mean squared displacement')
ax.set_title('Hotspots trap the particles and slow their spreading')
ax.legend()
fig.tight_layout()
plt.show()
