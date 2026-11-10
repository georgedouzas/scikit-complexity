"""
# Lotka-Volterra with harvesting

Prey and predators cycle out of phase as in the Lotka-Volterra system, but harvesting both with the same effort shifts
the centre of the cycle: on average there are more prey and fewer predators than without harvesting, Volterra's
principle.
"""

# %%
# The example builds `LotkaVolterraWithHarvestingSystem` with an effort of 0.4 on both species and runs it for about
# five cycles. The observer saves every second frame with the default blueprint, whose primary view is the time series
# of both populations.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import LotkaVolterraWithHarvestingSystem

SYSTEM = 'lotka_volterra_with_harvesting'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(LotkaVolterraWithHarvestingSystem(prey=16.0, predators=8.0), observers=[observer])

# %%
# Volterra's principle
# --------------------
# The cycles run around `prey = (d + predator_harvest) / b` and `predators = (r - prey_harvest) / a`. Without
# harvesting the centre is `(d / b, r / a) = (10, 10)`, with it `(18, 6)`. The harvested orbit should circle the
# shifted centre.

harvested = trajectory
unharvested = simulate(
    LotkaVolterraWithHarvestingSystem(prey=16.0, predators=8.0, prey_harvest=0.0, predator_harvest=0.0),
    until=50,
    dt=0.1,
)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(
    unharvested.get_values('prey.x'),
    unharvested.get_values('predators.x'),
    color='tab:grey',
    label='without harvesting',
)
ax.plot(10.0, 10.0, 'o', color='tab:grey')
ax.plot(harvested.get_values('prey.x'), harvested.get_values('predators.x'), color='tab:blue', label='with harvesting')
ax.plot(18.0, 6.0, 'o', color='tab:blue')
ax.set_xlabel('prey')
ax.set_ylabel('predators')
ax.set_title('Harvesting moves the centre to more prey, fewer predators')
ax.legend()
plt.show()
