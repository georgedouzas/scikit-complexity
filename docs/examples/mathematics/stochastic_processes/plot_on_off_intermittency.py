"""
# Stochastic on-off intermittency

At the bifurcation point, each variable rests near zero for long laminar phases of very different durations, broken
by short bursts in which it rises to values of order one and falls back.
"""

# %%
# The system is created by name from the catalog with two independent realizations at the bifurcation point
# `epsilon = 0`, with noise intensity `D = 1/2`, and simulated for 600 time units in the Stratonovich convention. The
# observer saves every frame with the default blueprint, whose primary view holds the two variables.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import DistributionAnalysis, EventAnalysis
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'on_off_intermittency'

system = create_system(SYSTEM, n_realizations=2, epsilon=0.0, D=0.5)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=2.0,
    solver=EulerMaruyamaSolver(dt=0.01),
    observers=[observer],
    random_state=0,
).prepare()
simulation.run(until=598)

# %%
# Durations of the laminar phases
# -------------------------------
# While `x` is small, `y = ln x` performs a free random walk, and the nonlinearity pushes it back once `x` reaches
# order one, so `y` behaves as a random walk with a reflecting wall near 0. A laminar phase is an excursion of this
# walk away from the wall, and the first return times of a random walk are distributed as `tau^(-3/2)`. The durations
# of the phases below `x = 0.05` in 50 realizations should follow this power law over two decades, and `ln x` of the
# run above should wander far below 0 and never far above it.

ensemble = Simulation(
    create_system(SYSTEM, n_realizations=50),
    dt=0.1,
    solver=EulerMaruyamaSolver(dt=0.02),
    random_state=1,
).prepare()
values = ensemble.run(until=500).trajectory_.values('variables.x')
step = float(np.median(np.diff(ensemble.trajectory_.times)))
events = EventAnalysis(threshold=-0.05)
durations = step * np.concatenate(
    [events.prepare(-values[:, index]).run().durations_[1:-1] for index in range(values.shape[1])],
)
histogram = DistributionAnalysis('histogram', bins=12, log=True).prepare(durations[durations >= 0.5]).run()
edges, density = histogram.edges_, histogram.density_
centers = np.sqrt(edges[1:] * edges[:-1])

fig, (laminar, logarithm) = plt.subplots(1, 2, figsize=(10, 4))
laminar.loglog(centers, density, 'o', label='simulated')
closest = np.argmin(np.abs(centers - 1.0))
laminar.loglog(centers, density[closest] * (centers / centers[closest]) ** -1.5, 'k--', label='tau^(-3/2)')
laminar.set_xlabel('duration of a laminar phase')
laminar.set_ylabel('probability density')
laminar.set_title('Laminar phases have power-law durations')
laminar.legend()
logarithm.plot(simulation.trajectory_.times, np.log(simulation.trajectory_.values('variables.x')[:, 0]))
logarithm.axhline(0.0, color='grey', linestyle='--')
logarithm.set_xlabel('time')
logarithm.set_ylabel('ln x')
logarithm.set_title('ln x is a random walk below a wall')
fig.tight_layout()
plt.show()
