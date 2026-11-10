"""
# Stochastic on-off intermittency

At the bifurcation point, each variable rests near zero for long laminar phases of very different durations, broken
by short bursts in which it rises to values of order one and falls back.
"""

# %%
# The example builds `OnOffIntermittencySystem` with two independent realizations at the bifurcation point
# `epsilon = 0`, with noise intensity `D = 1/2`, and simulates it for 600 time units in the Stratonovich convention. The
# observer saves every frame with the default blueprint, whose primary view holds the two variables.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import distribution, events
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.stochastic_processes import OnOffIntermittencySystem

SYSTEM = 'on_off_intermittency'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(
    OnOffIntermittencySystem(n_realizations=2, epsilon=0.0, D=0.5),
    until=598,
    dt=2.0,
    solver=EulerMaruyamaSolver(dt=0.01),
    observers=[observer],
    random_state=0,
)

# %%
# Durations of the laminar phases
# -------------------------------
# While `x` is small, `y = ln x` performs a free random walk, and the nonlinearity pushes it back once `x` reaches
# order one, so `y` behaves as a random walk with a reflecting wall near 0. A laminar phase is an excursion of this
# walk away from the wall, and the first return times of a random walk are distributed as `tau^(-3/2)`. The durations
# of the phases below `x = 0.05` in 50 realizations should follow this power law over two decades, and `ln x` of the
# run above should wander far below 0 and never far above it.

ensemble = simulate(
    OnOffIntermittencySystem(n_realizations=50),
    until=500,
    dt=0.1,
    solver=EulerMaruyamaSolver(dt=0.02),
    random_state=1,
)
values = ensemble.get_values('variables.x')
step = float(np.median(np.diff(ensemble.times)))
durations = step * np.concatenate(
    [events(-values[:, index], threshold=-0.05).durations[1:-1] for index in range(values.shape[1])],
)
histogram = distribution(durations[durations >= 0.5], kind='histogram', bins=12, log=True)
edges, density = histogram.edges, histogram.density
centers = np.sqrt(edges[1:] * edges[:-1])

fig, (laminar, logarithm) = plt.subplots(1, 2, figsize=(10, 4))
laminar.loglog(centers, density, 'o', label='simulated')
closest = np.argmin(np.abs(centers - 1.0))
laminar.loglog(centers, density[closest] * (centers / centers[closest]) ** -1.5, 'k--', label='tau^(-3/2)')
laminar.set_xlabel('duration of a laminar phase')
laminar.set_ylabel('probability density')
laminar.set_title('Laminar phases have power-law durations')
laminar.legend()
logarithm.plot(trajectory.times, np.log(trajectory.get_values('variables.x')[:, 0]))
logarithm.axhline(0.0, color='grey', linestyle='--')
logarithm.set_xlabel('time')
logarithm.set_ylabel('ln x')
logarithm.set_title('ln x is a random walk below a wall')
fig.tight_layout()
plt.show()
