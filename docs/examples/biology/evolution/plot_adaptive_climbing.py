"""
# Adaptive climbing

A population under strong selection climbs a random fitness landscape one beneficial mutation at a time. Each step up
is harder to find than the last, so the walk slows near the top and stops on a local optimum.
"""

# %%
# Ten individuals with genomes of twenty sites mutate with probability 0.1. The default blueprint shows the mean
# fitness of the population climbing in steps.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import AdaptiveClimbingSystem

SYSTEM = 'adaptive_climbing'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(
    AdaptiveClimbingSystem(size=10, u=0.1, length=20, seed=0),
    dt=1,
    observers=[observer],
    random_state=0,
)

# %%
# An adaptive walk
# ----------------
# The best fitness in the population should rise in steps that grow ever rarer, the signature of an adaptive walk
# towards a local optimum. Averaging over several landscapes smooths the steps into a decelerating curve.

best = []
for seed in range(8):
    trajectory = simulate(
        AdaptiveClimbingSystem(size=10, u=0.1, length=20, seed=seed),
        until=200,
        dt=1,
        random_state=seed,
    )
    best.append(
        [trajectory.get_agents('population', frame).values['fitness'].max() for frame in range(len(trajectory.times))],
    )
best = np.array(best)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trajectory.times, best.T, color='tab:blue', linewidth=0.6, alpha=0.6)
ax.plot(trajectory.times, best.mean(axis=0), color='black', label='mean of eight landscapes')
ax.set_xlabel('generation')
ax.set_ylabel('best fitness')
ax.set_title('Fitness rises in ever rarer steps')
ax.legend()
plt.show()
