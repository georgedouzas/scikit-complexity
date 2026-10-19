"""
# Stochastic on-off intermittency

At the bifurcation point, each variable rests near zero for long laminar phases of very different durations, broken
by short bursts in which it rises to values of order one and falls back.
"""

# %%
# The model is created by name from the catalog with two independent realizations at the bifurcation point
# `epsilon = 0`, with noise intensity `D = 1/2`, and simulated for 600 time units in the Stratonovich convention. The
# observer saves every frame with the default blueprint, whose primary view holds the two variables.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import detect_events, histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'on_off_intermittency'

system = create(MODEL, n_realizations=2, epsilon=0.0, D=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=2.0, solver=EulerMaruyama(dt=0.01), observers=[observer], random_state=0)
simulation.run(until=598)

# %%
# Durations of the laminar phases
# -------------------------------
# While `x` is small, `y = ln x` performs a free random walk, and the nonlinearity pushes it back once `x` reaches
# order one, so `y` behaves as a random walk with a reflecting wall near 0. A laminar phase is an excursion of this
# walk away from the wall, and the first return times of a random walk are distributed as `tau^(-3/2)`. The durations
# of the phases below `x = 0.05` in 50 realizations should follow this power law over two decades, and `ln x` of the
# run above should wander far below 0 and never far above it.

ensemble = Simulation(create(MODEL, n_realizations=50), dt=0.1, solver=EulerMaruyama(dt=0.02), random_state=1)
values = ensemble.run(until=500).trajectory_['variables.x']
durations = np.concatenate(
    [
        detect_events(-values[:, index], threshold=-0.05, times=ensemble.trajectory_.times).durations[1:-1]
        for index in range(values.shape[1])
    ],
)
edges, density = histogram(durations[durations >= 0.5], bins=12, log=True)
centers = np.sqrt(edges[1:] * edges[:-1])

fig, (laminar, logarithm) = plt.subplots(1, 2, figsize=(10, 4))
laminar.loglog(centers, density, 'o', label='simulated')
closest = np.argmin(np.abs(centers - 1.0))
laminar.loglog(centers, density[closest] * (centers / centers[closest]) ** -1.5, 'k--', label='tau^(-3/2)')
laminar.set_xlabel('duration of a laminar phase')
laminar.set_ylabel('probability density')
laminar.set_title('Laminar phases have power-law durations')
laminar.legend()
logarithm.plot(simulation.trajectory_.times, np.log(simulation.trajectory_['variables.x'][:, 0]))
logarithm.axhline(0.0, color='grey', linestyle='--')
logarithm.set_xlabel('time')
logarithm.set_ylabel('ln x')
logarithm.set_title('ln x is a random walk below a wall')
fig.tight_layout()
plt.show()
