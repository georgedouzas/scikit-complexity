"""
# Size-structured forest

Trees grow in height, shade the trees below them, and compete for canopy space and resources, while their seeds
disperse to found saplings nearby. The density of trees over position and height follows a McKendrick-von Foerster
equation, where the light at a height is the exponential of the canopy integrated from that height to the top. On a
strip of forest whose surroundings were cleared, no seeds come from beyond the edges. The image shows the density over
position across the strip, down, and height, across.
"""

# %%
# The example builds `ForestSizeStructureSystem` on a strip of width 10 cut into 40 positions, with heights from the
# sapling height 0.1 to the asymptotic height 50 in 100 cells, a Hurst exponent of 1, and the dispersal length 1. The
# observer saves every frame of 5 up to `t = 200`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import ForestSizeStructureSystem

SYSTEM = 'forest_size_structure'
POSITIONS, HEIGHTS = 40, 100

system = ForestSizeStructureSystem(positions=POSITIONS, heights=HEIGHTS).prepare().system_
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='forest/x', interval=1)
trajectory = simulate(system, dt=5.0, solver=RK4Solver(dt=0.1), observers=[observer], until=200)

# %%
# Edges deplete small trees
# -------------------------
# Near the edges a site receives seeds only from one side, so fewer saplings enter. The small trees, whose numbers are
# set by recruitment, should be strongly depleted there, while the tall trees, whose numbers are set by competition for
# canopy space, should barely change. In the bulk, the tree-size distribution should fall as a power of the height,
# close to `h^-(1 + 2H) = h^-3` among the tall trees.

space = system.elements[0].space.prepare()
heights = space.coordinates_['h']
positions = (np.arange(POSITIONS) + 0.5) * 10 / POSITIONS
density = trajectory.get_values('forest.x')[-1].reshape(POSITIONS, HEIGHTS)
bulk, edge = density[POSITIONS // 2], density[0]
fig, (sizes, profile) = plt.subplots(1, 2, figsize=(11, 4))
sizes.loglog(heights, bulk, label='bulk')
sizes.loglog(heights, edge, label='edge')
sizes.loglog(heights, bulk[20] * (heights / heights[20]) ** -3.0, 'k--', label='h^-3')
sizes.set_xlabel('height h')
sizes.set_ylabel('density of trees')
sizes.set_title('Tree-size distributions')
sizes.legend()
total = density.sum(axis=1) * space.spacings_[1]
small = density[:, heights < 1.0].sum(axis=1)
tall = density[:, heights > 10.0].sum(axis=1)
profile.plot(positions, total / total[POSITIONS // 2], label='all trees')
profile.plot(positions, small / small[POSITIONS // 2], label='trees below 1')
profile.plot(positions, tall / tall[POSITIONS // 2], label='trees above 10')
profile.set_xlabel('position across the strip')
profile.set_ylabel('density relative to the bulk')
profile.set_title('The edges lose small trees')
profile.legend()
fig.tight_layout()
plt.show()
