"""
# Galton-Watson branching process

Every individual leaves a Poisson number of children, one per individual on average at criticality. Populations of
ten founders then wander up and down from generation to generation, and one after another they die out, even though
the mean size stays constant.
"""

# %%
# The model runs one population of ten founders with mean offspring `m = 1` for 40 generations and logs its size.
# Seven more independent populations are run alike, and the sizes of all eight are logged as time series after the run,
# and the blueprint puts them first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'galton_watson'


def sizes(trajectory):
    """Return the size of the population at every generation."""
    return np.array([len(trajectory.agents('population', frame).ids) for frame in range(len(trajectory.times))])


whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/sizes', name='population sizes', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, n_individuals=10), dt=1, observers=[observer], random_state=0).run(until=40)
runs = [simulation.trajectory_] + [
    Simulation(create(MODEL, n_individuals=10), dt=1, random_state=seed).run(until=40).trajectory_
    for seed in range(1, 8)
]
for index, run in enumerate(runs):
    observer.recording_.send_columns(
        f'sizes/population_{index}',
        indexes=[rr.TimeColumn('sim_time', duration=run.times)],
        columns=rr.Scalars.columns(scalars=sizes(run)),
    )
observer.recording_.flush()

# %%
# Extinction probability
# ----------------------
# With Poisson offspring the generating function is `f(s) = exp(m (s - 1))`. Starting from one individual, the
# probability to be extinct by generation `n` is the `n`-th iterate of `f` from 0, and the ultimate extinction
# probability is the smallest root of `q = f(q)`: 1 up to `m = 1` and below 1 above it. The share of 200 runs extinct
# after 12 generations should follow the 12-th iterate.

rates = np.linspace(0.5, 2.0, 7)
extinct = [
    np.mean(
        [
            len(
                Simulation(create(MODEL, n_individuals=1, m=m), dt=1, random_state=seed)
                .run(until=12)
                .trajectory_.agents('population', 12)
                .ids,
            )
            == 0
            for seed in range(200)
        ],
    )
    for m in rates
]
grid = np.linspace(0.5, 2.0, 100)
iterate, ultimate = np.zeros_like(grid), np.zeros_like(grid)
for _ in range(12):
    iterate = np.exp(grid * (iterate - 1))
for _ in range(5000):
    ultimate = np.exp(grid * (ultimate - 1))
fig, (paths, extinction) = plt.subplots(1, 2, figsize=(11, 4))
for run in runs:
    paths.plot(run.times, sizes(run))
paths.set_xlabel('generation')
paths.set_ylabel('population size')
paths.set_title('Critical populations die out one by one')
extinction.plot(rates, extinct, 'o', label='simulated, 200 runs')
extinction.plot(grid, iterate, 'k--', label='extinct by generation 12')
extinction.plot(grid, ultimate, 'k:', label='ultimate extinction q')
extinction.set_xlabel('mean offspring m')
extinction.set_ylabel('extinction probability')
extinction.set_title('Extinction is certain up to m = 1')
extinction.legend()
fig.tight_layout()
plt.show()
