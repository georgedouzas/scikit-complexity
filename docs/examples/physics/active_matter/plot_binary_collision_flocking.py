"""
# Microscopic flocking with aligning binary collisions

The particles start at random headings. Every collision aligns the two partners, and the shared heading spreads until
most particles fly in one direction, so the points take one colour, which shows the heading.
"""

# %%
# The particles fly at unit speed on a torus of side 20 at the density 1, with self-diffusion events at unit rate,
# collisions at the distance 1/2, and the noise `sigma = 0.3`. The blueprint shows the whole torus and draws the
# particles larger than the model's visual does, so they read in the gallery thumbnail.

import math

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'binary_collision_flocking'

torus = rrb.Spatial2DView(
    origin='/birds',
    name='birds',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 20.0), y_range=(-20.0, 0.0)),
    overrides={'birds': rr.Points2D.from_fields(radii=0.25).visualizer()},
)
blueprint = rrb.Blueprint(
    torus,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, sigma=0.3, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=200)

# %%
# The order-disorder transition in noise
# --------------------------------------
# The hydrodynamic equations derived from the Boltzmann equation predict collective motion above the density
# `rho_t(sigma) = pi lam (1 - exp(-sigma^2 / 2)) / (4 (exp(-sigma^2 / 2) - 2/3))`. At the density 1 this puts the
# threshold noise near 0.64. The stationary polarization of short runs should be large at low noise and drop to the
# finite-size level, near `1 / sqrt(N)`, at a noise close to the predicted threshold. The theory is an expansion near
# the threshold, so the simulated transition sits somewhat below it.


def threshold_density(sigma: float) -> float:
    """Return the density above which the hydrodynamic theory predicts collective motion, for `lam = 1`."""
    return math.pi * (1 - math.exp(-(sigma**2) / 2)) / (4 * (math.exp(-(sigma**2) / 2) - 2 / 3))


threshold = brentq(lambda sigma: threshold_density(sigma) - 1.0, 0.05, 0.89)
noises = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
polarizations = []
for sigma in noises:
    run = Simulation(create(MODEL, sigma=float(sigma), random_state=1), dt=1, random_state=1).run(until=150)
    polarizations.append(np.mean(run.trajectory_['polarization'][75:]))
fig, (series, transition) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, simulation.trajectory_['polarization'])
series.set_ylim(0, 1)
series.set_xlabel('time')
series.set_ylabel('polarization')
series.set_title('Ordering at sigma = 0.3')
transition.plot(noises, polarizations, 'o-', label='simulation')
transition.axvline(threshold, color='black', linestyle='--', label=f'predicted threshold {threshold:.2f}')
transition.axhline(1 / math.sqrt(400), color='grey', linestyle=':', label='1 / sqrt(N)')
transition.set_ylim(0, 1)
transition.set_xlabel('noise sigma')
transition.set_ylabel('stationary polarization')
transition.set_title('Polarization against the noise at density 1')
transition.legend()
fig.tight_layout()
plt.show()
