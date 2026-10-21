"""
# Ideal gas

The molecules fly in straight lines and bounce off the walls of the box. They stay spread uniformly over the box, and
their speeds keep the Maxwell-Boltzmann distribution: slow (purple) and fast (yellow) molecules are mixed everywhere.
"""

# %%
# The molecules are drawn as points colored by speed. The blueprint draws them larger than the model's visual does, so
# they read in the gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'ideal_gas'

box = rrb.Spatial2DView(
    origin='/gas',
    name='gas',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(-0.02, 1.02), y_range=(-1.02, 0.02)),
    overrides={'gas': rr.Points2D.from_fields(radii=0.011).visualizer()},
)
blueprint = rrb.Blueprint(
    box,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, n_molecules=800, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=2.0)

# %%
# Speeds against the Maxwell-Boltzmann distribution
# -------------------------------------------------
# The temperature is measured from the kinetic energy of the simulated molecules, `kT = m <v^2> / 2` in two dimensions.
# The histogram of the speeds at the end of the run should follow the two-dimensional Maxwell-Boltzmann density
# `f(v) = (m v / kT) exp(-m v^2 / (2 kT))` at that temperature. Walls reverse velocity components without changing
# speeds, so the agreement holds at every frame.

velocities = simulation.trajectory_['gas.v'][-1]
speeds = np.linalg.norm(velocities, axis=1)
kt = float(np.mean(np.sum(velocities**2, axis=1))) / 2
edges, density = histogram(speeds, bins=25)
grid = np.linspace(0.0, speeds.max(), 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.stairs(density, edges, fill=True, alpha=0.5, label='simulated speeds')
ax.plot(grid, grid / kt * np.exp(-(grid**2) / (2 * kt)), color='black', label=f'Maxwell-Boltzmann, kT = {kt:.2f}')
ax.set_xlabel('speed')
ax.set_ylabel('probability density')
ax.set_title('Speeds of 800 molecules at the end of the run')
ax.legend()
plt.show()
