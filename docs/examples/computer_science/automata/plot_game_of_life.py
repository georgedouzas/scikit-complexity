"""
# Game of Life

From a random soup of live cells, most cells die within a few generations and the survivors settle into still lifes,
blinkers, and gliders scattered over a mostly empty board.
"""

# %%
# The board is a 100 by 100 lattice with three cells in ten alive at the start. The image shows live cells in yellow.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'game_of_life'

system = create(MODEL, shape=(100, 100), density=0.3, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).run(until=200)

# %%
# Live cells over time
# --------------------
# The density of live cells should fall quickly from its initial 0.3 and level off near a few percent, once the
# board holds only still lifes, oscillators, and gliders.

alive = simulation.trajectory_.lattice('cells.alive')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(simulation.trajectory_.times, alive.mean(axis=(1, 2)))
ax.set_xlabel('generation')
ax.set_ylabel('density of live cells')
ax.set_title('The soup decays and settles')
plt.show()
