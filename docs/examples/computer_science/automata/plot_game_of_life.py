"""
# Game of Life

From a random soup of live cells, most cells die within a few generations and the survivors settle into still lifes,
blinkers, and gliders scattered over a mostly empty board.
"""

# %%
# The board is a 100 by 100 lattice with three cells in ten alive at the start. The image shows live cells in yellow.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import GameOfLifeSystem

SYSTEM = 'game_of_life'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(GameOfLifeSystem(shape=(100, 100), density=0.3, random_state=0), until=200, observers=[observer])

# %%
# Live cells over time
# --------------------
# The density of live cells should fall quickly from its initial 0.3 and level off near a few percent, once the
# board holds only still lifes, oscillators, and gliders.

alive = trajectory.get_lattice('cells.alive')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(trajectory.times, alive.mean(axis=(1, 2)))
ax.set_xlabel('generation')
ax.set_ylabel('density of live cells')
ax.set_title('The soup decays and settles')
plt.show()
