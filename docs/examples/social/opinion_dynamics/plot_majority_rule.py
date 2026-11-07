"""
# Majority rule

Every voter on a lattice takes the majority opinion of its Moore neighbourhood, itself included. Isolated dissenters
convert at once, so the random salt-and-pepper start coarsens within a few steps into smooth domains of agreement,
which then freeze, because a flat boundary has a majority on each side. Across the whole lattice the rule amplifies
whichever opinion starts in the majority.
"""

# %%
# On a 100 by 100 lattice the voters hold opinion 1 with probability one half. The blueprint shows the opinions after
# 20 synchronous steps, frozen into domains.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import domains
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'majority_rule'
STEPS = 20

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = (
    Simulation(create_system(SYSTEM, density=0.5, random_state=0), observers=[observer]).prepare().run(until=STEPS)
)

# %%
# Coarsening, freezing, and amplification
# ---------------------------------------
# The interface density, the share of neighbouring pairs that disagree, should fall from its random value of one half
# and stop within a few steps as the domains freeze. Repeated over initial densities, the final share of opinion 1
# should follow an S-shaped curve steeper than the diagonal: a majority of 55 percent grows to a larger one, and a
# minority shrinks.

grids = simulation.trajectory_.lattice('voters.opinion')
densities = np.linspace(0.3, 0.7, 17)
finals = [
    Simulation(create_system(SYSTEM, density=density, random_state=0))
    .prepare()
    .run(until=STEPS)
    .trajectory_.values('voters.opinion')[-1]
    .mean()
    for density in densities
]
fig, (interfaces, amplification) = plt.subplots(1, 2, figsize=(10, 4))
interface = domains(grids, kind='interface-density', times=simulation.trajectory_.times)
interfaces.plot(interface.times, interface.density, 'o-', ms=3)
interfaces.set_xlabel('step')
interfaces.set_ylabel('interface density')
interfaces.set_title('Domains form and freeze')
amplification.plot(densities, finals, 'o-', label='after 20 steps')
amplification.plot([0.3, 0.7], [0.3, 0.7], 'k:', label='initial share')
amplification.set_xlabel('initial share of opinion 1')
amplification.set_ylabel('final share of opinion 1')
amplification.set_title('The majority is amplified')
amplification.legend()
fig.tight_layout()
plt.show()
