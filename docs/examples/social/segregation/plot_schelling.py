"""
# Schelling segregation

Schelling's model of residential segregation: residents of two kinds live on a grid, and a resident moves to a
random empty cell whenever fewer than a threshold share of its neighbours are of its own kind. Even residents who
are content to be in a minority sort themselves into large homogeneous neighbourhoods, a macro pattern far more
segregated than anyone's preference.
"""

# %%
# On a 50 by 50 grid, 80 percent of the cells hold residents, half of each kind, who want at least 40 percent of their
# neighbours to be like them. The blueprint shows the residents after 50 steps, sorted into large patches.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import like_neighbor_share
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'schelling'
SIDE, STEPS = 50, 50

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
system = create(MODEL, width=SIDE, height=SIDE, minority=0.5, threshold=0.4, random_state=0)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=STEPS)

# %%
# Mild preferences, strong segregation
# ------------------------------------
# The like-neighbour share, the average share of a resident's neighbours that are of its kind, starts near one half
# for random placement. It should climb within a few dozen steps far above the threshold that residents ask for, and
# across thresholds the final share should exceed the threshold for every mild preference, Schelling's tipping. Very
# demanding residents never settle, and their segregation falls back as they keep moving.


def like_share(trajectory, frame):
    """Return the like-neighbour share of the residents at a frame."""
    residents = trajectory.agents('residents', frame)
    grid = np.full((SIDE, SIDE), -1.0)
    columns, rows = np.asarray(residents.locations, dtype=int).T
    grid[rows, columns] = residents.values['kind']
    return like_neighbor_share(grid, empty=-1.0, diagonal=True)


thresholds = np.linspace(0.1, 0.8, 8)
finals = []
for threshold in thresholds:
    system = create(MODEL, width=SIDE, height=SIDE, minority=0.5, threshold=threshold, random_state=0)
    trajectory = Simulation(system, random_state=0).run(until=STEPS).trajectory_
    finals.append(like_share(trajectory, STEPS))
trajectory = simulation.trajectory_
fig, (series, sweep) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times, [like_share(trajectory, frame) for frame in range(STEPS + 1)])
series.axhline(0.4, color='grey', linestyle=':', label='threshold 0.4')
series.set_xlabel('step')
series.set_ylabel('like-neighbour share')
series.set_title('Segregation emerges')
series.legend()
sweep.plot(thresholds, finals, 'o-', label='after 50 steps')
sweep.plot([0.1, 0.8], [0.1, 0.8], 'k:', label='threshold')
sweep.set_xlabel('threshold of similar neighbours')
sweep.set_ylabel('like-neighbour share')
sweep.set_title('Segregation exceeds the preference')
sweep.legend()
fig.tight_layout()
plt.show()
