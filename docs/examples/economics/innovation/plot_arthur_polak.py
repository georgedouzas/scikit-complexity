"""
# Arthur-Polak technology evolution

Arthur and Polak's computer experiment on the evolution of technology: starting from a NAND gate and the constants,
random circuits are wired from the existing technologies, and a circuit is kept when it computes a useful logic
function. Each invention becomes a building block for the next, so simple functions such as negation appear first and
complex ones, which would be hopeless to find from NAND gates alone, follow in bursts as their components become
available.
"""

# %%
# The pool starts with three technologies and tries one random circuit of up to twelve components and three inputs
# per step, for 3000 steps. The blueprint shows the size of the pool, which grows in rare jumps, sampled every twenty
# steps.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'arthur_polak'
STEPS, SEEDS = 3000, 12

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters',
        contents=['/reporters/technologies.count'],
        name='technologies',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=20)
simulation = Simulation(create(MODEL), observers=[observer], random_state=0).run(until=STEPS)

# %%
# Complexity and the time of invention
# ------------------------------------
# The cost of a technology is the number of NAND gates it expands to. Over twelve runs, the step at which each
# invention enters the pool should grow with its cost: cheap functions are found within a few hundred steps, and the
# expensive three-input functions only after the intermediate inventions they are built from.

fig, (pools, invention) = plt.subplots(1, 2, figsize=(10, 4))
for seed in range(SEEDS):
    trajectory = Simulation(create(MODEL), random_state=seed).run(until=STEPS).trajectory_
    count = trajectory['technologies.count']
    pools.step(trajectory.times, count, where='post', color='C0', alpha=0.4)
    costs = trajectory.agents('technologies', len(trajectory.times) - 1).values['cost']
    found = [int(np.argmax(count > index)) for index in range(3, int(count[-1]))]
    invention.loglog(costs[3:], np.maximum(found, 1), 'o', color='C1', alpha=0.6)
pools.set_xlabel('step')
pools.set_ylabel('technologies in the pool')
pools.set_title(f'Pool growth in {SEEDS} runs')
invention.set_xlabel('cost of the invention (NAND gates)')
invention.set_ylabel('step of invention')
invention.set_title('Complex inventions come later')
fig.tight_layout()
plt.show()
