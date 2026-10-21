"""
# Social impact theory

Every person feels the persuasion of those who disagree and the support of those who agree, weakened with distance, and
changes opinion when the net pressure to change is positive. The majority grows, but a minority can survive in compact
clusters, where its members support each other, and the grid freezes into a static pattern.
"""

# %%
# The model is created by name from the catalog on a 15 by 15 grid with half of the people of each opinion, random
# persuasiveness and supportiveness, the distance exponent 2, and no noise, and runs for 20 steps. The default
# blueprint shows the grid, coloured by opinion.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'social_impact'

system = create(MODEL, random_state=2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer], random_state=0).run(until=20)

# %%
# Consensus or a surviving minority
# ---------------------------------
# Without noise the dynamics should stop changing within a few steps. Depending on the start, the minority either
# vanishes, a consensus, or survives as clusters that hold a stable share, and a steeper decay of impact with distance
# should protect the clusters more often.

fig, ax = plt.subplots(figsize=(7, 4.5))
for exponent, style in [(2.0, '-'), (3.0, '--')]:
    for seed in range(6):
        run = Simulation(create(MODEL, exponent=exponent, random_state=seed), random_state=0).run(until=20)
        trajectory = run.trajectory_
        shares = [
            np.mean(np.asarray(trajectory.agents('people', frame).values['opinion']) > 0)
            for frame in range(len(trajectory.times))
        ]
        ax.plot(
            trajectory.times,
            shares,
            style,
            color=f'C{seed}',
            label=f'exponent {exponent:g}' if seed == 0 else None,
        )
ax.set_xlabel('step')
ax.set_ylabel('share of opinion +1')
ax.set_title('Six starts for each exponent of the decay of impact')
ax.legend()
plt.show()
