"""
# Bonabeau hierarchies

Agents that wander on a grid fight whenever they meet, and winning makes the next win more likely. At high density the
fights are frequent enough to outrun the slow decay of strength, so a few agents become dominant and many become
subordinate, while sparse agents stay alike.
"""

# %%
# Half of the cells of a 20 x 20 torus hold an agent. The default blueprint colors each agent by its strength, on a
# range that widens as the strengths spread, so the dominant agents stand out in yellow and the subordinate ones in
# purple.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bonabeau_hierarchies'

system = create(MODEL, density=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=100)

# %%
# Spread of strengths against density
# -----------------------------------
# Every fight moves one unit of strength from the loser to the winner, and every step removes a tenth of each strength.
# The more often agents meet, the further the strengths spread before the decay pulls them back, so the standard
# deviation of the strengths after 100 steps should grow with the density, from about 1 for sparse agents to about
# 10 when most cells are full.

densities = np.linspace(0.1, 0.8, 8)
spreads = []
for density in densities:
    runs = [
        Simulation(create(MODEL, density=density, random_state=seed), random_state=seed).run(until=100)
        for seed in range(3)
    ]
    finals = [run.trajectory_.agents('contenders', len(run.trajectory_) - 1).values['strength'] for run in runs]
    spreads.append([np.std(final) for final in finals])
spreads = np.asarray(spreads)
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(densities, spreads.mean(axis=1), yerr=spreads.std(axis=1), marker='o', capsize=3)
ax.set_xlabel('density of agents')
ax.set_ylabel('standard deviation of strength')
ax.set_title('Denser crowds build steeper hierarchies')
plt.show()
