"""
# Life in a tree

The cells sit on a ball of radius 4 of the free group F4, a tree in which every cell has eight neighbours along `a`,
`b`, `c`, `d`, and their inverses. A cell is alive after a step when the sum over itself and its neighbours is 3, or
4 with the cell alive, Conway's rule on the Cayley graph of F4. On the tree the automaton is surjective but not
pre-injective.
"""

# %%
# The ball holds 3201 elements on rings by word length, colored by their state.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'life_in_a_tree'

system = create_system(SYSTEM, radius=4, density=0.3, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).prepare().run(until=20)

# %%
# Density by word length and a collision
# --------------------------------------
# Most cells of the ball lie on its boundary, where only one neighbour is inside, so they die at once, and the dying
# front moves inward one ring per step until the truncated ball is empty. A single live cell dies and leaves its
# neighbours dead, so it has the same image as the empty configuration, two configurations that differ at one cell, the
# failure of pre-injectivity on the whole tree.

space = system.elements[0].space.prepare()
alive = simulation.trajectory_.values('cells.alive')
lone = np.zeros(space.size_)
lone[0] = 1.0
images = [
    Simulation(create_system(SYSTEM, radius=4, initial=cells)).prepare().run(until=1).trajectory_.values('cells.alive')
    for cells in (lone, np.zeros(space.size_))
]
fig, ax = plt.subplots(figsize=(7, 4))
for length in range(5):
    ax.plot(
        simulation.trajectory_.times, alive[:, space.lengths_ == length].mean(axis=1), 'o-', label=f'length {length}',
    )
ax.set_xlabel('step')
ax.set_ylabel('density of live cells')
ax.set_title(f'A lone live cell and the empty tree share their image: {np.array_equal(images[0][-1], images[1][-1])}')
ax.legend()
plt.show()
