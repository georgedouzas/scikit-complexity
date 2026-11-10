"""
# Stigmergic predator-prey

Plants, prey, and predators share a grid. Animals mark their happiness and distress on the ground, and the marks of
their own kind draw them to good places and away from bad ones. The prey graze the plants bare and boom, then starve,
while the predators grow on the prey and decline after them.
"""

# %%
# The example builds `StigmergicPredatorPreySystem` with 300 plants, 120 prey, and 20 predators on a 30 by 30 grid. The
# observer saves every step with the default blueprint, whose primary view is the image of the marks.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import StigmergicPredatorPreySystem

SYSTEM = 'stigmergic_predator_prey'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(StigmergicPredatorPreySystem(random_state=0), observers=[observer], random_state=0)

# %%
# Populations with and without shared emotions
# --------------------------------------------
# The three populations should peak in turn, plants, then prey, then predators. Turning the shared emotions off
# changes how the animals move, and with it the timing of the prey boom and crash.

without = simulate(StigmergicPredatorPreySystem(emotions=False, random_state=0), random_state=0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, run, title in zip(axes, [trajectory, without], ['with emotions', 'without'], strict=True):
    for kind in ['plants', 'prey', 'predators']:
        ax.plot(run.times, run.get_values(f'life/{kind}.count'), label=kind)
    ax.set_xlabel('step')
    ax.set_title(title)
axes[0].set_ylabel('organisms')
axes[0].legend()
fig.tight_layout()
plt.show()
