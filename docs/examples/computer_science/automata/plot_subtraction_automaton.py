"""
# Subtraction automaton over the naturals

The cells sit on a segment of 41 cells over the natural numbers, and every cell becomes `max(0, x(n) - x(n + 1))`.
The staircase `x_M`, equal to `M` before the origin and descending by one from it, maps to a block of `M` ones from the
origin. As `M` grows these blocks converge to the ones on the whole half line, a configuration that is no image, so
the image of the automaton over the infinite alphabet `N` is not closed.
"""

# %%
# The run starts from the staircase of height 10 and shows the cells at every step.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import SubtractionAutomatonSystem

SYSTEM = 'subtraction_automaton'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(SubtractionAutomatonSystem(size=41, height=10), observers=[observer])

# %%
# Images of the staircases
# ------------------------
# The image of every staircase `x_M` should be the block of `M` ones from the origin, so the images agree with the
# half line of ones on longer and longer windows. A preimage of the half line itself would need a cell `x(n)` larger
# than `x(n + 1)` by one for ever, a strictly decreasing sequence of naturals, which does not exist.

heights = [2, 5, 10, 20]
fig, ax = plt.subplots(figsize=(7, 4))
for offset, height in enumerate(heights):
    staircase = SubtractionAutomatonSystem(size=41, height=height)
    image = simulate(staircase, until=1).get_values('cells.value')[-1]
    ax.step(np.arange(41) - 10, image + 1.5 * offset, where='mid', label=f'image of $x_{{{height}}}$')
ax.set_xlabel('cell n')
ax.set_yticks([])
ax.set_title('Blocks of ones grow towards the half line, which is no image')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
fig.tight_layout()
plt.show()
