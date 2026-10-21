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

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'adaptive_climbing'

system = create(MODEL, size=10, u=0.1, length=20, seed=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=200)

# %%
# An adaptive walk
# ----------------
# The best fitness in the population should rise in steps that grow ever rarer, the signature of an adaptive walk
# towards a local optimum. Averaging over several landscapes smooths the steps into a decelerating curve.

best = []
for seed in range(8):
    run = Simulation(create(MODEL, size=10, u=0.1, length=20, seed=seed), dt=1, random_state=seed).run(until=200)
    trajectory = run.trajectory_
    best.append(
        [trajectory.agents('population', frame).values['fitness'].max() for frame in range(len(trajectory.times))],
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
