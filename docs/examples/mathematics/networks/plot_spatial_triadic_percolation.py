"""
# Spatial triadic percolation

Nodes on a torus are linked to nearby nodes, and every link is regulated by nearby nodes, positively or negatively. A
link stays active only while one of its positive regulators lies in the giant component and none of its negative
regulators does. Since the regulation is local, the giant component forms clusters, stripes, and octopus-like shapes
that move and blink instead of settling.
"""

# %%
# Sixteen hundred nodes lie on a torus of side 4 with links of typical length 0.25 and regulators within a typical
# distance of 0.25, as positive as negative, and a link the regulation allows is active with probability `p = 0.8`.
# The nodes keep their positions, so the network view shows the giant component in space.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'spatial_triadic_percolation'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=100)
active = simulation.trajectory_.values('nodes.active').astype(float)
positions = system.elements[0].space.prepare().positions_

# %%
# Size of the giant component
# ---------------------------
# The share of nodes in the giant component should keep changing, alternating between larger and smaller values as
# the patterns blink.

fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(active.mean(axis=1), 'o-', markersize=3)
axis.set_xlabel('time step of Millán et al.')
axis.set_ylabel('share of nodes in the giant component')
axis.set_title('The giant component does not settle')
fig.tight_layout()
plt.show()

# %%
# Patterns
# --------
# The giant component of consecutive steps should cover complementary regions of the torus, a blinking pattern.

fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for axis, step in zip(axes, (60, 61, 62, 63), strict=True):
    axis.scatter(*positions.T, c=active[step], s=4, cmap='viridis', vmin=0, vmax=1)
    axis.set_title(f'step {step}')
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
fig.suptitle('Active nodes, the giant component, in yellow')
fig.tight_layout()
plt.show()
