"""
# Majority automaton on the Heisenberg group

The cells sit on a ball of radius 5 of the discrete Heisenberg group of upper unitriangular integer matrices. Each cell
takes the majority value of itself and its right neighbours along `a`, `a^-1`, `b`, and `b^-1`, counting only the
neighbours inside the ball, so the random start freezes within a few steps into large clusters of agreeing cells.
"""

# %%
# The ball's elements sit on rings by word length, colored by their state.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'heisenberg_automaton'

system = create(MODEL, radius=5, density=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).run(until=20)

# %%
# Polynomial growth and freezing
# ------------------------------
# The Heisenberg group is nilpotent, and its balls grow as the fourth power of the radius, by the Bass-Guivarc'h
# formula. The local log-log slope of the ball sizes should rise towards 4 from below, and the fraction of cells that
# change at each step should fall to 0 as the majority rule freezes.

space = system.elements[0].space
sizes = np.asarray(space.ball_sizes(), dtype=float)
alive = simulation.trajectory_['cells.alive']
changed = (np.diff(alive, axis=0) != 0).mean(axis=1)
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
left.loglog(np.arange(1, sizes.size), sizes[1:], 'o-', label='ball size')
left.loglog(np.arange(1, sizes.size), np.arange(1, sizes.size) ** 4.0, '--', label='$n^4$')
left.set_xlabel('radius $n$')
left.set_ylabel('$|B(n)|$')
left.set_title('Growth of the Heisenberg group')
left.legend()
right.plot(np.arange(1, alive.shape[0]), changed, 'o-')
right.set_xlabel('step')
right.set_ylabel('fraction of cells that change')
right.set_title('The majority rule freezes')
plt.show()
