"""
# Cubic chaotic map

Two hundred chaotic walkers start at 0 and spread out like random walkers. Every step is `2x - 1` for the current value
of the cubic map `x' = 15 x (x - 1/2)(x - 1) + x`, so the walk is deterministic and still fans out diffusively.
"""

# %%
# The walkers start with values of `x` spread regularly over the unit interval, as in Bertin's Figure 7.4, and walk for
# 250 steps. The model logs the positions `y` of all walkers, and the blueprint shows them over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'cubic_chaotic_map'

system = create(MODEL, n_walkers=200, lam=15.0)
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/walkers/y', name='positions y', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1, observers=[observer], observables=describe(MODEL).observables_of(system)).run(
    until=250,
)

# %%
# Diffusion of the chaotic walk and the density of the map
# --------------------------------------------------------
# The mean position should stay near 0 and the mean square position should grow linearly in time, as for a random walk.
# The slope is twice the diffusion coefficient, which the Green-Kubo sum `C_0 + 2 sum_k C_k` of the autocovariances
# `C_k` of the steps, measured over the ensemble up to lag 20, predicts. The negative `C_1` of anticorrelated steps
# makes the slope about a third of the step variance `C_0`.
# The histogram of `x` should be far from uniform, with peaks near both ends, as in Bertin's Figure 7.5.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
steps = 2 * trajectory['walkers.x'][50:] - 1
covariances = [np.mean(steps[: len(steps) - lag] * steps[lag:]) for lag in range(21)]
predicted = covariances[0] + 2 * sum(covariances[1:])
fig, (spread, density) = plt.subplots(1, 2, figsize=(10, 4))
spread.plot(times, trajectory['square_displacement'], label=r'$\langle y_t^2 \rangle$')
spread.plot(times, trajectory['displacement'], label=r'$\langle y_t \rangle$')
spread.plot(times, predicted * times, '--', color='black', label='Green-Kubo slope')
spread.set_xlabel('step t')
spread.set_ylabel('moment of the position')
spread.set_title('Linear spreading without drift')
spread.legend()
density.hist(trajectory['walkers.x'][50:].ravel(), bins=50, range=(0, 1), density=True)
density.set_xlabel('x')
density.set_ylabel('density')
density.set_title('Invariant density of the cubic map, λ = 15')
fig.tight_layout()
plt.show()
