"""
# Naming game

Agents with empty inventories invent names for one object and teach them to each other in pairs. A success wipes both
inventories down to the agreed word, a failure adds the word to the hearer's inventory. Inventories first swell with
competing words, then collapse onto a single shared name.
"""

# %%
# Fifty agents play one game each per step on average. The default blueprint shows the mean number of words in an
# inventory, which climbs above two as invented words spread and then falls to exactly one at consensus.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'naming_game'

system = create(MODEL, n_agents=50)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=40)

# %%
# Time to consensus
# -----------------
# Consensus is reached when every inventory holds the same single word. On the complete graph the number of games
# until consensus should grow as `N^1.5`, a slope of 1.5 on logarithmic axes, since every step plays about `N` games
# and the number of steps grows as `N^0.5`.


def consensus_games(n_agents, seed):
    """Return the number of games, steps times agents, until every inventory holds the same single word."""
    run = Simulation(create(MODEL, n_agents=n_agents), random_state=seed).run(until=200)
    trajectory = run.trajectory_
    for frame in range(1, len(trajectory)):
        words = trajectory.agents('namers', frame).values['words']
        if np.all(words.sum(axis=1) == 1) and np.count_nonzero(words.sum(axis=0)) == 1:
            return frame * n_agents
    return np.nan


sizes = np.array([25, 50, 100, 200])
games = np.array([[consensus_games(size, seed) for seed in range(3)] for size in sizes])
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(sizes, np.nanmean(games, axis=1), 'o', label='simulated, mean of 3 runs')
ax.loglog(sizes, np.nanmean(games[0]) * (sizes / sizes[0]) ** 1.5, '--', color='black', label='slope 1.5')
ax.set_xlabel('number of agents N')
ax.set_ylabel('games until consensus')
ax.set_title('Consensus takes about N^1.5 games')
ax.legend()
plt.show()
