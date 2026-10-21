"""
# Drossel-Schwabl forest fire

Trees grow slowly on empty sites, lightning strikes very rarely, and a burning tree sets its four neighbours on fire.
Growth fills the forest between fires and each strike burns a whole connected patch, so the forest settles into a
patchwork of stands of every age. The view shows the cells of a 200 x 200 lattice, empty,
tree, or burning, with fire fronts sweeping through the older, denser stands.
"""

# %%
# The model is created by name from the catalog with growth probability `0.02` and lightning probability `1e-5`
# per step. The observer saves every second step with the default blueprint, whose primary view shows the cells.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'forest_fire'

system = create(MODEL, shape=(200, 200), growth=0.02, lightning=1e-5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=598)

# %%
# Steady density of trees
# -----------------------
# A forest denser than the site percolation threshold `0.593` holds a spanning cluster that a single strike burns
# down, so after a transient of large boom-and-bust cycles the density of trees should hover around a steady value
# well below that threshold, held there by the balance of growth and burning, with fluctuations from the fires.

steady = Simulation(create(MODEL, shape=(128, 128), growth=0.02, lightning=1e-5), dt=1, random_state=1).run(5000)
trees = (steady.trajectory_['forest.cover'] == 1).mean(axis=1)

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
