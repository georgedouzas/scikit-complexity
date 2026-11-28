"""
# Drossel-Schwabl forest fire

Trees grow slowly on empty sites, lightning strikes very rarely, and a burning tree sets its four neighbours on fire.
Growth fills the forest between fires and each strike burns a whole connected patch, so the forest settles into a
patchwork of stands of every age. The view shows the cells of a 200 x 200 lattice, empty,
tree, or burning, with fire fronts sweeping through the older, denser stands.
"""

# %%
# The example builds `ForestFireSystem` with growth probability `0.02` and lightning probability `1e-5`
# per step. The observer saves every second step with the default blueprint, whose primary view shows the cells.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.criticality import ForestFireSystem

SYSTEM = 'forest_fire'

system = ForestFireSystem(shape=(200, 200), growth=0.02, lightning=1e-5, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = system.prepare().simulate(until=598, observers=[observer], random_state=0)

# %%
# Steady density of trees
# -----------------------
# A forest denser than the site percolation threshold `0.593` holds a spanning cluster that a single strike burns
# down, so after a transient of large boom-and-bust cycles the density of trees should hover around a steady value
# well below that threshold, held there by the balance of growth and burning, with fluctuations from the fires.

steady = ForestFireSystem(shape=(128, 128), growth=0.02, lightning=1e-5).prepare().simulate(until=5000, random_state=1)
trees = (steady.get_values('forest.cover') == 1).mean(axis=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trees, label='simulation, 128 x 128')
ax.axhline(trees[1000:].mean(), color='black', label=f'steady mean {trees[1000:].mean():.2f}')
ax.axhline(0.593, color='gray', linestyle='--', label='percolation threshold 0.593')
ax.set_xlabel('step')
ax.set_ylabel('density of trees')
ax.set_title('The forest settles below the percolation threshold')
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Sizes of fires
# --------------
# Every fire is a `fire` event of the run, the burning trees connected in space and time, with its size, the number of
# trees that burn. Rare lightning lets the forest grow large stands between strikes, so the sizes of the fires should
# spread over several decades.

sizes = np.asarray([event.size for event in steady.events if event.time > 1000])
histogram = distribution(sizes, kind='histogram', bins=20, log=True)
centers = np.sqrt(histogram.edges[1:] * histogram.edges[:-1])

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(centers[histogram.density > 0], histogram.density[histogram.density > 0], 'o')
ax.set_xlabel('size of a fire, trees')
ax.set_ylabel('density')
ax.set_title(f'{sizes.size} fires of the steady forest')
fig.tight_layout()
plt.show()
