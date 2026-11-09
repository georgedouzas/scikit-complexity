"""
# Involutive automaton

The cells sit on a ring of 80 cells and hold 0 or 1, and every cell reads the window from its left neighbour to the
`n`-th cell on its right. A cell at the left end of a run of exactly `n` ones becomes 0, a cell just before a run of
exactly `n - 1` ones becomes 1, and every other cell keeps its value. The two moves undo each other, so the automaton
is its own inverse.
"""

# %%
# The space-time image shows the ring at every step for `n = 3`, alternating between two configurations.

import itertools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import InvolutiveAutomatonSystem

SYSTEM = 'involutive_automaton'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(InvolutiveAutomatonSystem(n=3, size=80, density=0.6, random_state=0), observers=[observer])

# %%
# Period two
# ----------
# Every configuration should come back after two steps, so the distance to the start alternates between a positive
# value and 0, and over all rings of 12 cells the second step returns every one of them.

alive = trajectory.get_values('cells.alive')
distance = (alive != alive[0]).sum(axis=1)
configurations = np.asarray(list(itertools.product((0, 1), repeat=12)), dtype=float)
returned = [
    np.array_equal(
        simulate(InvolutiveAutomatonSystem(n=3, initial=configuration), until=2).get_values(
            'cells.alive',
        )[-1],
        configuration,
    )
    for configuration in configurations[::17]
]
fig, (image, check) = plt.subplots(1, 2, figsize=(10, 4))
image.imshow(alive, aspect='auto', cmap='Greys', interpolation='nearest')
image.set_xlabel('cell n')
image.set_ylabel('step')
image.set_title('The automaton alternates')
check.plot(trajectory.times, distance, 'o-')
check.set_xlabel('step')
check.set_ylabel('cells that differ from the start')
check.set_title(f'Two steps return every sampled ring: {all(returned)}')
fig.tight_layout()
plt.show()
