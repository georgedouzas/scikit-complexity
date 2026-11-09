"""
# Kinetic crowd in a street

Bellomo's kinetic model of a crowd: walkers first choose a direction, weighing their target, the mean stream, and the
search of less crowded areas by the density and their stress, and then adjust their speed to the density they
perceive ahead. Walkers near a wall turn parallel to it. Two groups walk in opposite directions along a street, and
panic raises the stress.
"""

# %%
# Fifty walkers, half walking each way, start uniformly in a street of length 20, periodic along it, and width 5,
# with the stress `beta = 0.5`, which rises to 0.9 at step 150. The default blueprint shows the walkers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.crowds import KineticCrowdSystem

SYSTEM = 'kinetic_crowd'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
system = KineticCrowdSystem(panic_time=150, panic_stress=0.9)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# The fundamental diagram and the effect of panic
# -----------------------------------------------
# With one cell over the street the crowd is homogeneous, and the mean speed should follow the fundamental diagram of
# Eq. 6.30, faster for anxious walkers in good venues. In the street the mean speed should jump when the stress rises.


def diagram(density, gamma, deceleration=0.95):
    """Return the mean speed of the fundamental diagram of Eq. 6.30."""
    free = gamma**2 * (1 - density) ** 2
    return gamma * free / (free + (1 - deceleration * gamma) * density**2)


fig, (fundamental, panic) = plt.subplots(1, 2, figsize=(10, 4))
grid = np.linspace(0, 1, 100)
for gamma in (0.4, 0.8):
    walkers = np.array([40, 120, 200, 280, 360])
    speeds = []
    for count in walkers:
        homogeneous = KineticCrowdSystem(n_walkers=int(count), stress=gamma, cell=20.0)
        run = simulate(homogeneous, random_state=0)
        speeds.append(run.get_values('decisions.mean_speed')[-100:].mean())
    fundamental.plot(walkers / 400, speeds, 'o', label=f'simulation, alpha beta = {gamma}')
    fundamental.plot(grid, diagram(grid, gamma), color='black', linestyle=':')
fundamental.set_xlabel('density')
fundamental.set_ylabel('mean speed')
fundamental.set_title('Fundamental diagram')
fundamental.legend()
panic.plot(trajectory.times[1:], trajectory.get_values('decisions.mean_speed')[1:], label='mean speed')
panic.plot(trajectory.times[1:], trajectory.get_values('decisions.band_index')[1:], label='band index')
panic.axvline(150, color='black', linestyle=':', label='panic')
panic.set_xlabel('steps')
panic.set_title('Counterflow in the street')
panic.legend()
fig.tight_layout()
plt.show()
