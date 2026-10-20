"""
# Lotka-Volterra with harvesting

Prey and predators cycle out of phase as in the Lotka-Volterra model, but harvesting both with the same effort shifts
the centre of the cycle: on average there are more prey and fewer predators than without harvesting, Volterra's
principle.
"""

# %%
# The model is created by name from the catalog with an effort of 0.4 on both species and run for about five cycles.
# The observer saves every second frame with the default blueprint, whose primary view is the time series of both
# populations.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'lotka_volterra_with_harvesting'

system = create(MODEL, prey=16.0, predators=8.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=50)

# %%
# Volterra's principle
# --------------------
# The cycles run around `prey = (d + predator_harvest) / b` and `predators = (r - prey_harvest) / a`. Without
# harvesting the centre is `(d / b, r / a) = (10, 10)`, with it `(18, 6)`. The harvested orbit should circle the
# shifted centre.

harvested = simulation.trajectory_
unharvested = Simulation(create(MODEL, prey=16.0, predators=8.0, prey_harvest=0.0, predator_harvest=0.0), dt=0.1)
unharvested = unharvested.run(until=50).trajectory_
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(unharvested['prey.n'], unharvested['predators.n'], color='tab:grey', label='without harvesting')
ax.plot(10.0, 10.0, 'o', color='tab:grey')
ax.plot(harvested['prey.n'], harvested['predators.n'], color='tab:blue', label='with harvesting')
ax.plot(18.0, 6.0, 'o', color='tab:blue')
ax.set_xlabel('prey')
ax.set_ylabel('predators')
ax.set_title('Harvesting moves the centre to more prey, fewer predators')
ax.legend()
plt.show()
