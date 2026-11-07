"""
# Squirmers

A squirmer is a sphere that swims by a tangential slip of its surface, the model of ciliated microorganisms such as
Volvox and Paramecium of Lighthill and Blake. The first slip mode `B1` sets its speed `2 B1 / 3`, and the second,
`B2 = beta B1`, how it stirs the fluid: a pusher (`beta < 0`) pushes fluid out along its axis and draws it in from the
sides, and a puller (`beta > 0`) does the opposite. The view shows 27 pushers in an unbounded fluid, each moved and
turned by the flows of all the others.
"""

# %%
# The system is built from its class: 27 squirmers with `B1 = 1.5` and `beta = -3` on a jittered cubic
# lattice of spacing 5 radii, with random orientations. The flows of the other squirmers enter through the mobility of
# `MobilityInteraction`, so the run is a set of ordinary differential equations for the positions and orientations,
# which the default solver integrates. The observer saves every frame of 0.5 with the default blueprint.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import SquirmerElement, SquirmerSlipInteraction, SquirmersSystem

SYSTEM = 'squirmers'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = simulate(SquirmersSystem(), observers=[observer])

# %%
# Pushers attract side by side, pullers repel
# -------------------------------------------
# Two squirmers that swim side by side at a distance `r` feel only the stresslet of the other, whose flow across the
# axis is `-B2 a^2 / (2 r^2)` outwards (Ishikawa, Simmonds, and Pedley 2006). So two pushers should close their gap,
# two pullers should open it, and two neutral squirmers (`beta = 0`), whose flows decay as `1 / r^3` along the
# swimming direction only, should keep it.

fig, ax = plt.subplots(figsize=(7, 4))
for beta, color in ((-3.0, 'tab:blue'), (0.0, 'tab:gray'), (3.0, 'tab:orange')):
    pair = GroupElement(
        'squirmers',
        element=SquirmerElement,
        size=2,
        values={'x': [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], 'p': [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]},
    )
    slip = SquirmerSlipInteraction('slip', target='squirmers', B1=1.5, B2=1.5 * beta)
    system = System(elements=[pair], interactions=[slip], space=ContinuousSpace(n_dim=3))
    trajectory = Simulation(system, dt=0.25).prepare().run(until=12).trajectory_
    positions = trajectory.values('squirmers.x')
    gap = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
    kind = 'pushers' if beta < 0 else 'pullers' if beta > 0 else 'neutral squirmers'
    ax.plot(trajectory.times, gap, color=color, label=f'{kind}, beta = {beta:.0f}')
ax.axhline(2.0, color='black', linestyle=':', label='contact')
ax.set_xlabel('time')
ax.set_ylabel('distance between the centres')
ax.set_title('Two squirmers swimming side by side')
ax.legend()
fig.tight_layout()
plt.show()
