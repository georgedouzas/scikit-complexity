"""
# Vicsek system

Particles move at one speed and turn towards the mean heading of their neighbours, with a little noise. From random
headings they align into a flock that crosses the box together: the points take one color, which shows the heading.
Variants align with the nearest neighbours only, see only within a vision cone, or react to delayed headings.
"""

# %%
# Four hundred particles move `v0 = 0.5` per step in a periodic box of side 10, at density 4 and noise `eta = 0.1`.
# The blueprint shows the whole box and draws the particles larger than the system's visual does, so they read in the
# gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import VicsekSystem

SYSTEM = 'vicsek'

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
system = VicsekSystem(n_particles=400, eta=0.1, seed=0)
observables = VicsekSystem.observables
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, until=150, observers=[observer], random_state=0, observables=observables)

# %%
# Order against noise
# -------------------
# The Vicsek system orders at low noise and disorders at high noise. The stationary polar order parameter
# `|<exp(i theta)>|` should fall from near one at small `eta` towards the finite-size level `1 / sqrt(N)` as the noise
# grows, with the transition near `eta = 0.6` at this density.

noises = np.linspace(0.05, 0.95, 10)
orders = []
for noise in noises:
    system_system = VicsekSystem(n_particles=400, eta=noise, seed=1)
    run = simulate(
        system_system.prepare().system_,
        until=150,
        dt=5.0,
        random_state=1,
        observables=VicsekSystem.observables,
    )
    orders.append(np.mean(np.asarray(run.values('polarization'), dtype=float)[-10:]))
fig, (series, transition) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times, np.asarray(trajectory.values('polarization'), dtype=float))
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

# %%
# Nearest neighbours and delays
# -----------------------------
# Aligning with the `k` nearest particles at any distance, a sparse flock should order better as `k` grows, as in
# Fig. 8.10 of Argun et al. With a sensory delay, the particles react to the headings their neighbours had some steps
# ago, or to headings predicted ahead for a negative delay, and a long or negative delay should leave fewer neighbours
# around every particle than no delay, the clustering of Figs. 8.8 and 8.9.


def neighbours(positions, size):
    """Return the mean number of other particles within the unit distance."""
    gaps = np.abs(positions[:, None, :] - positions[None, :, :])
    distances = np.hypot(*np.moveaxis(np.minimum(gaps, size - gaps), -1, 0))
    return ((distances < 1).sum(axis=1) - 1).mean()


ks = [1, 2, 4, 8]
topological = []
for k in ks:
    system_system = VicsekSystem(n_particles=100, size=100.0, v0=1.0, neighbors=k, seed=0)
    run = simulate(
        system_system.prepare().system_,
        until=300,
        dt=10.0,
        random_state=0,
        observables=VicsekSystem.observables,
    )
    topological.append(np.mean(np.asarray(run.values('polarization'))[-5:]))
delays = [-4, -2, 0, 2, 5, 10, 25]
clustering = []
for delay in delays:
    system_system = VicsekSystem(n_particles=100, size=50.0, v0=0.15, eta=0.2 / np.pi, delay=delay, seed=0)
    positions = np.asarray(
        simulate(system_system, until=2000, dt=100.0, random_state=0).values('flock.x'),
    )
    clustering.append(np.mean([neighbours(frame, 50.0) for frame in positions[-5:]]))
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.plot(ks, topological, 'o-')
left.set_xlabel('nearest neighbours k')
left.set_ylabel('stationary polar order')
left.set_title('Topological alignment')
right.plot(delays, clustering, 'o-')
right.set_xlabel('delay h (negative: prediction)')
right.set_ylabel('neighbours within the unit distance')
right.set_title('Delayed alignment')
fig.tight_layout()
plt.show()
