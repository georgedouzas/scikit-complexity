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

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.organizations import BonabeauHierarchiesSystem

SYSTEM = 'bonabeau_hierarchies'

system = BonabeauHierarchiesSystem(density=0.5, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=100, observers=[observer], random_state=0)

# %%
# Spread of strengths against density
# -----------------------------------
# Every fight moves one unit of strength from the loser to the winner, and every step removes a tenth of each strength.
# The more often agents meet, the further the strengths spread before the decay pulls them back, so the standard
# deviation of the strengths after 100 steps should grow with the density, from about 1 for sparse agents to about
# 10 when most cells are full.

densities = np.linspace(0.1, 0.8, 8)
spread_runs = []
for density in densities:
    runs = [
        simulate(BonabeauHierarchiesSystem(density=density, random_state=seed), until=100, random_state=seed)
        for seed in range(3)
    ]
    finals = [run.agents('contenders', len(run.times) - 1).values['strength'] for run in runs]
    spread_runs.append([np.std(final) for final in finals])
spreads = np.asarray(spread_runs)
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(densities, spreads.mean(axis=1), yerr=spreads.std(axis=1), marker='o', capsize=3)
ax.set_xlabel('density of agents')
ax.set_ylabel('standard deviation of strength')
ax.set_title('Denser crowds build steeper hierarchies')
plt.show()
