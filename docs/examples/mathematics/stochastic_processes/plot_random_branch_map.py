"""
# Random two-branch interval map

At every step a coin decides which of two maps of the unit interval acts: `(1 + x) / 2`, which halves the distance to
1, or `mu x²`, which pulls towards 0. Neither branch alone is chaotic, yet the orbits never settle and fill a
stationary distribution whose shape the coin and `mu` set.
"""

# %%
# Two hundred orbits iterate 200 steps with `p1 = 0.5` and `mu = 1`, and the system logs their values as time series,
# which the blueprint shows.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.stochastic_processes import RandomBranchMapSystem

SYSTEM = 'random_branch_map'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/orbits', name='orbits', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(RandomBranchMapSystem(n_orbits=20), observers=[observer])

# %%
# Bifurcation diagram over the coin
# ---------------------------------
# For every probability `p1` the values of 300 orbits after 100 steps sample the stationary distribution (Dilão,
# Exercise 5.10). It should move from 0 at `p1 = 0` to 1 at `p1 = 1`, spread widest in between, and lie lower for a
# smaller `mu`, whose branch pulls harder towards 0. For `mu = 0` its mean is `p1 / (2 - p1)`. For `mu = 1` both
# branches fix `x = 1`, where the first halves the distance to 1 and the second doubles it, so the orbits collapse
# onto 1 once `p1` passes 1/2, a transition that the random choice of the branch alone makes.

probabilities = np.linspace(0.0, 1.0, 41)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for axis, mu in zip(axes, (0.0, 0.5, 1.0), strict=True):
    means = []
    for p1 in probabilities:
        run = simulate(RandomBranchMapSystem(n_orbits=300, p1=p1, mu=mu), until=100)
        x = run.get_values('orbits.x')[-1]
        axis.plot(np.full_like(x, p1), x, ',', color='C0')
        means.append(x.mean())
    axis.plot(probabilities, means, color='C3', label='mean')
    if mu == 0.0:
        axis.plot(probabilities, probabilities / (2 - probabilities), 'k:', label='p1 / (2 - p1)')
    if mu == 1.0:
        axis.axvline(0.5, color='black', linestyle=':', label='p1 = 1/2')
    axis.set_xlabel('probability p1')
    axis.set_title(f'mu = {mu}')
    axis.legend()
axes[0].set_ylabel('x after 100 steps')
fig.tight_layout()
plt.show()
