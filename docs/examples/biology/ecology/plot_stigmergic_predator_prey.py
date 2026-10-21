"""
# Stigmergic predator-prey

Plants, prey, and predators share a grid. Animals mark their happiness and distress on the ground, and the marks of
their own kind draw them to good places and away from bad ones. The prey graze the plants bare and boom, then starve,
while the predators grow on the prey and decline after them.
"""

# %%
# The model is created by name from the catalog with 300 plants, 120 prey, and 20 predators on a 30 by 30 grid. The
# observer saves every step with the default blueprint, whose primary view is the image of the marks.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stigmergic_predator_prey'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=100)

# %%
# Populations with and without shared emotions
# --------------------------------------------
# The three populations should peak in turn, plants, then prey, then predators. Turning the shared emotions off
# changes how the animals move, and with it the timing of the prey boom and crash.

without = Simulation(create(MODEL, emotions=False, random_state=0), dt=1, random_state=0).run(until=100).trajectory_
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, trajectory, title in zip(axes, [simulation.trajectory_, without], ['with emotions', 'without'], strict=True):
    for kind in ['plants', 'prey', 'predators']:
        ax.plot(trajectory.times, trajectory[f'life/{kind}.count'], label=kind)
    ax.set_xlabel('step')
    ax.set_title(title)
axes[0].set_ylabel('organisms')
axes[0].legend()
fig.tight_layout()
plt.show()
