"""
# Majority automaton on the free group

The cells sit on a ball of radius 6 of the free group of rank 2, a tree in which every cell has four neighbours. Each
cell takes the majority value of itself and its neighbours along `a`, `a^-1`, `b`, and `b^-1`, counting only the
neighbours inside the ball, so the random start freezes within a few steps into blocks of agreeing cells.
"""

# %%
# The ball holds 1457 elements on rings by word length, colored by their state.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'free_group_automaton'

system = create_system(SYSTEM, radius=6, density=0.5, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).prepare().run(until=20)

# %%
# Changes and agreement
# ---------------------
# The majority rule freezes, so the fraction of cells that change at a step should fall to 0 within a few steps, and
# the fraction of edges whose two cells agree should rise from one half, the value of the random start.

space = system.elements[0].space
alive = simulation.trajectory_.values('cells.alive')
edges = np.asarray([(first, second) for first, second, _ in space.edges_])
agreement = (alive[:, edges[:, 0]] == alive[:, edges[:, 1]]).mean(axis=1)
changed = (np.diff(alive, axis=0) != 0).mean(axis=1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(simulation.trajectory_.times, agreement, 'o-', label='edges whose cells agree')
ax.plot(simulation.trajectory_.times[1:], changed, 's-', label='cells that change')
ax.set_xlabel('step')
ax.set_ylabel('fraction')
ax.set_title('The majority rule freezes on the free group')
ax.legend()
plt.show()
