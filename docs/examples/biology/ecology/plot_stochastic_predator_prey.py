"""
# Stochastic predator-prey

Prey and predators born and killed one at a time keep fluctuating around their coexistence point instead of settling
on it: demographic noise excites the damped oscillations of the deterministic model into lasting, irregular cycles.
"""

# %%
# The model runs as a jump process of single births, deaths, and kills, with a carrying capacity of the prey below the
# value that destabilizes the coexistence point. The observer saves every frame with the default blueprint, whose
# primary view is the time series of both populations.

import matplotlib.pyplot as plt

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stochastic_predator_prey'

system = create(MODEL, stochastic=True)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer], random_state=0).prepare().run(until=100)

# %%
# Noise against the deterministic limit
# -------------------------------------
# The deterministic limit spirals into the coexistence point `x = m / (a (e - m h))`. In the phase plane the jump
# process should circle the same point without settling, its excursions as large as the first turns of the spiral.

noisy = simulation.trajectory_
deterministic = Simulation(create(MODEL, stochastic=False), dt=0.5).prepare().run(until=100).trajectory_
a, e, h, m = 0.005, 0.5, 0.5, 0.3
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(noisy.values('prey.n'), noisy.values('predators.n'), linewidth=0.8, label='jump process')
ax.plot(deterministic.values('prey.n'), deterministic.values('predators.n'), color='black', label='deterministic limit')
ax.axvline(m / (a * (e - m * h)), color='grey', linestyle=':', label='coexistence prey')
ax.set_xlabel('prey')
ax.set_ylabel('predators')
ax.set_title('Noise sustains the cycles the deterministic model damps')
ax.legend()
plt.show()
