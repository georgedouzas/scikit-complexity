"""
# Forest dieback

A tropical forest keeps its soil cool. When the ambient temperature rises, losing trees warms the soil, which slows the
growth of the remaining trees, and beyond a tipping temperature the forest dies back to bare soil.
"""

# %%
# The forest starts at the cover `v = 0.8` at the ambient temperature of 30 degrees and settles on its stable cover.
# The default blueprint shows the cover over 200 years.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import ForestDiebackSystem

SYSTEM = 'forest_dieback'

system = ForestDiebackSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(system, observers=[observer])

# %%
# Hysteresis of the cover
# -----------------------
# Every run starts from a dense forest, `v = 0.9`, or a sparse one, `v = 0.1`, and records the cover after 300 years.
# Between about 32.5 and 34.7 degrees the two starts should end differently, the dense forest surviving and the
# sparse one dying, a bistable range, while above it every forest should die.

temperatures = np.linspace(26.0, 36.0, 41)
fig, ax = plt.subplots(figsize=(6, 4))
for start, marker in [(0.9, 'o'), (0.1, 'x')]:
    final = [
        np.ravel(
            simulate(ForestDiebackSystem(v=start, Tv=t), until=300, dt=1.0).get_values('forest.v'),
        )[-1]
        for t in temperatures
    ]
    ax.plot(temperatures, final, marker, label=f'start at v = {start}')
ax.axvspan(32.5, 34.7, color='grey', alpha=0.2, label='bistable range')
ax.set_xlabel('ambient temperature (degrees Celsius)')
ax.set_ylabel('cover after 300 years')
ax.set_title('Forest dieback and its hysteresis')
ax.legend()
plt.show()
