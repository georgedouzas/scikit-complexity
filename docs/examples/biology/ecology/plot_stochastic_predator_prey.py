"""
# Stochastic predator-prey

Prey and predators born and killed one at a time keep fluctuating around their coexistence point instead of settling
on it: demographic noise excites the damped oscillations of the deterministic system into lasting, irregular cycles.
"""

# %%
# The system runs as a jump process of single births, deaths, and kills, with a carrying capacity of the prey below the
# value that destabilizes the coexistence point. The observer saves every frame with the default blueprint, whose
# primary view is the time series of both populations.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import StochasticPredatorPreySystem

SYSTEM = 'stochastic_predator_prey'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(StochasticPredatorPreySystem(stochastic=True), observers=[observer], random_state=0)

# %%
# Noise against the deterministic limit
# -------------------------------------
# The deterministic limit spirals into the coexistence point `x = m / (a (e - m h))`. In the phase plane the jump
# process should circle the same point without settling, its excursions as large as the first turns of the spiral.

noisy = trajectory
deterministic = simulate(StochasticPredatorPreySystem(stochastic=False))
a, e, h, m = 0.005, 0.5, 0.5, 0.3
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(noisy.values('prey.x'), noisy.values('predators.x'), linewidth=0.8, label='jump process')
ax.plot(deterministic.values('prey.x'), deterministic.values('predators.x'), color='black', label='deterministic limit')
ax.axvline(m / (a * (e - m * h)), color='grey', linestyle=':', label='coexistence prey')
ax.set_xlabel('prey')
ax.set_ylabel('predators')
ax.set_title('Noise sustains the cycles the deterministic system damps')
ax.legend()
plt.show()
