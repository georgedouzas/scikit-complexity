"""
# Probabilistic Boolean network

Three genes switch on and off as every gene draws one of its candidate functions at every step. From the same start,
some runs end with all genes on and others with all genes off, the two attractors of the network.
"""

# %%
# The network follows the three-gene example of Shmulevich et al., in which `x1` and `x3` have two candidate functions
# and `x2` one, and starts with `x1` and `x3` on. In this run the network alternates between `101` and `110` before it
# falls into the attractor `000`. The blueprint shows the three genes over 30 steps.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.compile import compile_system
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'probabilistic_boolean_network'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/',
        contents=['/x1/x', '/x2/x', '/x3/x'],
        name='genes',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, active=('x1', 'x3')), dt=1, observers=[observer], random_state=4).run(30)

# %%
# Attractors reached from one start
# ---------------------------------
# The states `000` and `111` are fixed by every candidate function, so every run ends in one of them. Iterating the
# compiled update of the network 2000 times from the start `101` shows the share of runs in each attractor over the
# steps, which should settle near the absorption probabilities 1/3 for `000` and 2/3 for `111` that the probabilities
# of the tables give, and the share of runs in which `x1` used its first function, which should be its probability
# 0.6 at every step.

problem = compile_system(create(MODEL, active=('x1', 'x3')))
rng = np.random.default_rng(0)
names = problem.states.names
values = [names.index(f'{gene}.x') for gene in ('x1', 'x2', 'x3')]
states = np.tile(problem.initial_state, (2000, 1))
shares, first = [], []
for step in range(15):
    states = np.array([problem.update(float(step), state, problem.initial_parameters, rng) for state in states])
    codes = states[:, values] @ np.array([4, 2, 1])
    shares.append([np.mean(codes == 0), np.mean(codes == 7)])
    first.append(np.mean(states[:, names.index('x1.function')] == 0))
fig, (attractors, functions) = plt.subplots(1, 2, figsize=(10, 4))
attractors.plot(range(1, 16), np.array(shares), 'o-', label=['all off, 000', 'all on, 111'])
attractors.axhline(1 / 3, linestyle='--', color='black', label='absorption probabilities')
attractors.axhline(2 / 3, linestyle='--', color='black')
attractors.set_xlabel('step')
attractors.set_ylabel('share of runs')
attractors.set_title('Runs from 101 end in two attractors')
attractors.legend()
functions.plot(range(1, 16), first, 'o-', label='simulation')
functions.axhline(0.6, linestyle='--', color='black', label='probability 0.6')
functions.set_ylim(0.5, 0.7)
functions.set_xlabel('step')
functions.set_ylabel('share of runs')
functions.set_title('x1 uses its first function')
functions.legend()
fig.tight_layout()
plt.show()
