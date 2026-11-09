"""
# Lotka-Volterra

The prey and predator populations cycle out of phase: every peak of the prey is followed by a peak of the predators,
which then collapse until the prey recover, and the cycle repeats with the same amplitude.
"""

# %%
# The system is built from its class, with initial populations away from the equilibrium of four prey and 2.75
# predators, and simulated for five cycles. The observer saves every second frame of the run with the default
# blueprint, whose primary view is one time series view of both populations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

SYSTEM = 'lotka_volterra'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(LotkaVolterraSystem(prey=10.0, predators=5.0), observers=[observer])

# %%
# Phase portrait and the conserved quantity
# -----------------------------------------
# The system `dx/dt = r x - a x y`, `dy/dt = b x y - d y` conserves `V = b x - d ln x + a y - r ln y`. The trajectory
# should be one closed orbit around the equilibrium `(d / b, r / a)`, and `V` should stay constant up to the error
# of the solver.

prey, predators = trajectory.values('prey.x'), trajectory.values('predators.x')
r, a, b, d = 1.1, 0.4, 0.1, 0.4
conserved = b * prey - d * np.log(prey) + a * predators - r * np.log(predators)
fig, (orbit, drift) = plt.subplots(1, 2, figsize=(10, 4))
orbit.plot(prey, predators)
orbit.plot(d / b, r / a, 'o', color='black', label='equilibrium')
orbit.set_xlabel('prey')
orbit.set_ylabel('predators')
orbit.set_title('Phase portrait: one closed orbit')
orbit.legend()
drift.plot(trajectory.times, (conserved - conserved[0]) / abs(conserved[0]))
drift.set_xlabel('time')
drift.set_ylabel('relative change of V')
drift.set_title('The conserved quantity stays constant')
fig.tight_layout()
plt.show()
