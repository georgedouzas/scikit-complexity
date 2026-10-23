"""
# Ginelli nematic particles

Self-propelled particles that align with the axes of their neighbours, not their directions, come to share one axis but
move both ways along it: the points split into two colors, the two opposite headings along the common axis.
"""

# %%
# Four hundred particles move `v0 = 0.3` per step in a periodic box of side 10 with noise `eta = 0.08`. The blueprint
# shows the whole box and draws the particles larger than the model's visual does, so they read in the gallery
# thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'ginelli_nematics'

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
system = create(MODEL, n_particles=400, seed=0)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0, observables=observables).run(until=150)

# %%
# Nematic and polar order
# -----------------------
# Nematic alignment orders the axes, so the nematic order parameter `|<exp(2 i theta)>|` should approach one while the
# polar order parameter `|<exp(i theta)>|` stays near the finite-size level `1 / sqrt(N)`, since as many particles move
# one way as the other. Both should fall as the noise grows.

noises = np.linspace(0.04, 0.4, 10)
nematic, polar = [], []
for noise in noises:
    model_system = create(MODEL, n_particles=400, eta=noise, seed=1)
    run = Simulation(model_system, dt=5.0, random_state=1, observables=describe(MODEL).observables_of(model_system))
    trajectory = run.run(until=150).trajectory_
    nematic.append(np.mean(np.asarray(trajectory['nematic_order'], dtype=float)[-10:]))
    polar.append(np.mean(np.asarray(trajectory['polarization'], dtype=float)[-10:]))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(noises, nematic, 'o-', label='nematic order')
ax.plot(noises, polar, 's-', label='polar order')
ax.axhline(1 / np.sqrt(400), color='grey', linestyle=':', label='1 / sqrt(N)')
ax.set_ylim(0, 1)
ax.set_xlabel('noise eta')
ax.set_ylabel('stationary order')
ax.set_title('Nematic order without polar order')
ax.legend()
plt.show()
