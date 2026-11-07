"""
# Iterated prisoner's dilemma on a lattice

Argun's evolutionary game: players on a lattice play a seven-round prisoner's dilemma with their four neighbours,
counting years in prison. A strategy cooperates for a number of rounds and then defects, and the best reply to a
strategy is to defect one round earlier. Players copy the best strategy around them and mutate now and then, so
cooperative and defecting strategies chase each other across the lattice.
"""

# %%
# A 30 by 30 lattice with every strategy from 0 to 7 runs for 200 steps at `R = 0.6`, `S = 1.5`, and a mutation rate
# of 1%. The default blueprint shows the strategies on the lattice.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'iterated_lattice_dilemma'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), observers=[observer], random_state=0).prepare().run(until=200)

# %%
# Cycles of strategies and the regimes of two strategies
# ------------------------------------------------------
# With every strategy the shares should rise and fall in turn, as in Fig. 13.7. With only always-cooperate and
# always-defect, cooperation should dominate at `R = 0.82`, coexist at `R = 0.84`, and vanish at `R = 0.86`, as in
# Exercise 13.3.

strategies = simulation.trajectory_.values('players.strategy').reshape(201, -1)
shares = np.array([np.bincount(row.astype(int), minlength=8) / row.size for row in strategies])
fig, (cycles, regimes) = plt.subplots(1, 2, figsize=(10, 4))
cycles.stackplot(simulation.trajectory_.times, shares.T, labels=[f'n = {n}' for n in range(8)])
cycles.set_xlabel('steps')
cycles.set_ylabel('share of the strategy')
cycles.set_title('Strategies emerge, grow, and vanish')
cycles.legend(fontsize='small', ncol=2)
for reward in (0.82, 0.84, 0.86):
    system = create_system(SYSTEM, reward=reward, extremes=True)
    values = Simulation(system, random_state=0).prepare().run(100).trajectory_.values('players.strategy')
    regimes.plot(np.mean(values.reshape(101, -1) == 7, axis=1), label=f'R = {reward}')
regimes.set_xlabel('steps')
regimes.set_ylabel('share of cooperators')
regimes.set_title('Two strategies, three regimes')
regimes.legend()
fig.tight_layout()
plt.show()
