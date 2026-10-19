"""
# Lotka-Volterra

The prey and predator populations cycle out of phase: every peak of the prey is followed by a peak of the predators,
which then collapse until the prey recover, and the cycle repeats with the same amplitude.
"""

# %%
# The model is created by name from the catalog, with initial populations away from the equilibrium of four prey and
# 2.75 predators, and simulated for five cycles. The observer saves every second frame of the run with the default
# blueprint, whose primary view is one time series view of both populations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'lotka_volterra'

system = create(MODEL, prey=10.0, predators=5.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=50)

# %%
# Phase portrait and the conserved quantity
# -----------------------------------------
# The model `dx/dt = r x - a x y`, `dy/dt = b x y - d y` conserves `V = b x - d ln x + a y - r ln y`. The trajectory
# should be one closed orbit around the equilibrium `(d / b, r / a)`, and `V` should stay constant up to the error
# of the solver.

prey, predators = simulation.trajectory_['prey.n'], simulation.trajectory_['predators.n']
r, a, b, d = 1.1, 0.4, 0.1, 0.4
conserved = b * prey - d * np.log(prey) + a * predators - r * np.log(predators)
fig, (orbit, drift) = plt.subplots(1, 2, figsize=(10, 4))
orbit.plot(prey, predators)
orbit.plot(d / b, r / a, 'o', color='black', label='equilibrium')
orbit.set_xlabel('prey')
orbit.set_ylabel('predators')
orbit.set_title('Phase portrait: one closed orbit')
orbit.legend()
drift.plot(simulation.trajectory_.times, (conserved - conserved[0]) / abs(conserved[0]))
drift.set_xlabel('time')
drift.set_ylabel('relative change of V')
drift.set_title('The conserved quantity stays constant')
fig.tight_layout()
plt.show()
