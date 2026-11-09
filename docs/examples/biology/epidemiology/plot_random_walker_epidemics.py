"""
# Random walker epidemics

One sick walker among a thousand random walkers starts a deadly epidemic. The sick infect the walkers they meet and
die some steps later, so the disease spreads outwards as a front, like a forest fire, and burns out behind it.
"""

# %%
# The system is built from its class with a thousand walkers on a 64 by 64 grid. The observer saves every step with
# the default blueprint, whose primary view shows the walkers on the grid.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.epidemiology import RandomWalkerEpidemicsSystem

SYSTEM = 'random_walker_epidemics'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(
    RandomWalkerEpidemicsSystem(walkers=1000, random_state=0),
    dt=1,
    observers=[observer],
    random_state=0,
    until=299,
)

# %%
# Density threshold
# -----------------
# Above a critical density of walkers the epidemic should percolate and kill a large share of them, while below it
# the disease should die out after a few victims.


def dead_share(walkers, seed):
    """Return the share of walkers dead at the end of a run."""
    run = simulate(RandomWalkerEpidemicsSystem(walkers=walkers, random_state=seed), until=300, dt=1, random_state=seed)
    final = run.get_agents('walkers', len(run.times) - 1)
    return np.mean(final.values['health'] == 2)


counts = [100, 250, 500, 1000, 2000]
shares = [np.mean([dead_share(count, seed) for seed in range(3)]) for count in counts]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.array(counts) / 64**2, shares, 'o-')
ax.set_xscale('log')
ax.set_xlabel('walkers per cell')
ax.set_ylabel('share of walkers dead after 300 steps')
ax.set_title('Dense populations let the epidemic spread')
plt.show()
