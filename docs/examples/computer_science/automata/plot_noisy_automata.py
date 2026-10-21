"""
# Noisy elementary automaton

Rule 18 orders a random line into domains of a periodic pattern, separated by defects that wander like random walkers
and annihilate in pairs, so the defects thin out without end. Flipping a small share of the outputs at random creates
new defects as fast as they annihilate, and their density levels off at a value set by the noise.
"""

# %%
# The line has 200 cells, half of them alive at random, and rule 18 runs for 200 generations with a share 0.01 of
# flipped outputs. The space-time view stacks the generations from top to bottom.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'noisy_automata'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
system = create(MODEL, rule=18, noise=0.01, size=200)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=200)

# %%
# Annihilating defects
# --------------------
# In the ordered phase of rule 18 live cells are never neighbours, so pairs of adjacent live cells mark the defects.
# Without noise their density should decay as `t^(-1/2)`, as for random walkers that annihilate on contact. With
# noise it should follow the same decay at first and then level off, higher for more noise.

fig, ax = plt.subplots(figsize=(6, 4.5))
for noise in (0.0, 0.001, 0.01):
    run = Simulation(create(MODEL, rule=18, noise=noise, size=2000), dt=1, random_state=0).run(until=1000)
    alive = run.trajectory_.lattice('cells.alive')
    defects = (alive[:, :-1] * alive[:, 1:]).mean(axis=1)
    ax.loglog(run.trajectory_.times[1:], defects[1:], label=f'noise {noise}')
times = np.array([3, 1000])
ax.loglog(times, 0.03 * (times / 3) ** -0.5, 'k--', label=r'$t^{-1/2}$')
ax.set_xlabel('generation t')
ax.set_ylabel('density of adjacent live pairs')
ax.set_title('Defects annihilate; noise keeps creating them')
ax.legend()
plt.show()
