"""
# Discrete master equation

Probability flows between 50 states on a ring, each with the potential `U = 2 cos(2 pi i / 50)`, by moves to the two
neighbouring states with Glauber probabilities that satisfy detailed balance. Started entirely on the state of highest
potential, the probability spreads both ways around the ring, slides into the valley opposite, and settles on the
Boltzmann distribution. The space-time image shows the probabilities, one row per saved step with the earliest at the
top.
"""

# %%
# The system is built from its class at `beta = 1` with attempt probability `w = 0.5`. The observer saves
# every step with the default blueprint, whose primary view is the space-time image of the probabilities.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.statistical_mechanics import DiscreteMasterEquationSystem

SYSTEM = 'discrete_master_equation'

system = DiscreteMasterEquationSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=250, dt=1, observers=[observer])

# %%
# Relaxation to the Boltzmann distribution
# ----------------------------------------
# Detailed balance makes `exp(-beta U) / Z` stationary, and the H-theorem makes the Kullback-Leibler divergence of the
# probabilities from it fall at every step. For two temperatures the divergence should fall monotonically towards
# zero, and exponentially at late times, set by the slowest mode of the transition matrix.


def divergence(beta):
    """Return the Kullback-Leibler divergence from the Boltzmann distribution at every step."""
    run = simulate(DiscreteMasterEquationSystem(beta=beta), until=2000, dt=1)
    probabilities = np.clip(run.get_values('states.P'), 1e-300, None)
    boltzmann = np.exp(-beta * run.get_values('states.U')[0])
    boltzmann /= boltzmann.sum()
    return run.times, np.sum(probabilities * np.log(probabilities / boltzmann), axis=1)


fig, ax = plt.subplots(figsize=(7, 4))
for beta in (0.5, 1.0):
    times, values = divergence(beta)
    ax.semilogy(times, values, label=f'beta = {beta}')
ax.set_xlabel('step')
ax.set_ylabel('KL divergence from Boltzmann')
ax.set_title('The master equation relaxes monotonically to Boltzmann')
ax.legend()
fig.tight_layout()
plt.show()
