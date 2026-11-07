"""
# Fredkin's replicator

Every cell of this planar automaton takes the parity of its four nearest neighbours. The rule is so simple that it is
linear modulo 2, and that makes it copy any pattern: at the times 2, 4, 8, 16, and so on, the initial pattern reappears
as four translated copies, and in between the copies multiply into more intricate arrangements.
"""

# %%
# The torus has 128 by 128 cells and starts from a circumference of radius 4, as in Dilão's example. The run lasts 32
# steps and the image shows live cells in yellow.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'fredkin_replicator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run()

# %%
# Copies at the powers of two
# ---------------------------
# The snapshots should show the circle at step 0, a crowded pattern at steps 13 and 26, and exactly four circles at
# steps 16 and 32, each shifted by that many cells in one of the four directions.

cells = simulation.trajectory_.lattice('cells.alive')
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
for ax, step in zip(axes, (0, 13, 16, 26, 32), strict=True):
    ax.imshow(np.asarray(cells[step]), cmap='viridis')
    ax.set_title(f'n = {step}')
    ax.set_axis_off()
plt.show()
