"""
# Nilpotent automaton of class n

The cells sit on a ring of 80 cells and hold 0 or 1. A cell reads the window from its left neighbour to the `n - 1`
cells on its right, and becomes 0 when the whole window holds ones or when it is the left end of a run of ones,
keeping its value otherwise. After one step every run of ones is shorter than `n`, and every further step removes the
left end of each run, so `n` steps send every configuration to 0.
"""

# %%
# The space-time image shows the ring at every step for the class `n = 4`, the oldest at the top.

import itertools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import NilpotentAutomatonSystem

SYSTEM = 'nilpotent_automaton'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(NilpotentAutomatonSystem(n=4, size=80, density=0.7, random_state=0), observers=[observer])

# %%
# Nilpotency class
# ----------------
# Over a sample of the rings of 12 cells, the largest number of steps before the zero configuration should be exactly
# `n`, the nilpotency class, reached by a run of `n + 1` ones.

classes = range(2, 7)
configurations = np.asarray(list(itertools.product((0, 1), repeat=12)), dtype=float)
largest = []
for n in classes:
    steps = []
    for configuration in configurations[::41]:
        values = simulate(NilpotentAutomatonSystem(n=n, initial=configuration), until=n)
        alive = values.get_values('cells.alive').any(axis=1)
        steps.append(int(np.argmin(alive)) if not alive[-1] else n + 1)
    largest.append(max(steps))
fig, (image, check) = plt.subplots(1, 2, figsize=(10, 4))
image.imshow(trajectory.get_values('cells.alive'), aspect='auto', cmap='Greys', interpolation='nearest')
image.set_xlabel('cell n')
image.set_ylabel('step')
image.set_title('Runs of ones lose their left ends')
check.plot(list(classes), largest, 'o', label='largest number of steps to 0')
check.plot(list(classes), list(classes), '--', color='black', label='class n')
check.set_xlabel('n')
check.legend()
check.set_title('Every ring reaches 0 in n steps')
fig.tight_layout()
plt.show()
