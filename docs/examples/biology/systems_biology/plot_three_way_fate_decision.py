"""
# Three-way fate decision

Cells descend the compactified elliptic umbilic, the minimal landscape of a decision among three fates. Without a
signal cells near the centre split among three attractors, and a drifting signal crosses a fold curve, where fates
disappear and their cells switch to the one that remains.
"""

# %%
# The system is built from its class with thirty cells near the centre and no signal, and the observer saves every
# frame with the default blueprint, whose primary view is the time series of the cell states.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import ThreeWayFateDecisionSystem

SYSTEM = 'three_way_fate_decision'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ThreeWayFateDecisionSystem(), observers=[observer])

# %%
# Landscape and decisions
# -----------------------
# The cells should end in the three minima of the potential, and under a drifting signal `theta1` in only one.

grid = np.linspace(-1.5, 1.5, 200)
x, y = np.meshgrid(grid, grid)
potential = x**4 + y**4 + x**3 - 2 * x * y**2 - 0.2 * (x**2 + y**2)
drifted = simulate(ThreeWayFateDecisionSystem(v1=0.05))
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.contour(x, y, potential, levels=30, cmap='Greys')
ax.plot(trajectory.values('cells.x'), trajectory.values('cells.y'), color='tab:blue', lw=0.5)
ax.plot(trajectory.values('cells.x')[-1], trajectory.values('cells.y')[-1], 'o', color='tab:blue', label='no signal')
ax.plot(drifted.values('cells.x')[-1], drifted.values('cells.y')[-1], 's', color='tab:red', label='drifting signal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Three fates, and one left after the signal')
ax.legend()
plt.show()
