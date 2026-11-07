"""
# TAP growth and autocatalytic sets

Hordijk and Kauffman's combination of the theory of the adjacent possible with autocatalytic sets: goods are made from
combinations of existing goods, so their number grows slowly for a long time and then explodes, the hockey stick of
the TAP equation. Every good catalyses the production of other goods at random, and once the catalysis is frequent
enough the production functions form an autocatalytic set, a RAF, that tends to hold most of the economy.
"""

# %%
# From 10 goods, new goods of up to four parents appear at the rates `alpha^i C(M, i)` with `alpha = 0.01`, until the
# economy holds 1250 goods. The production function of every good is a hyperedge of its parents and the good, and
# the catalysis is a directed edge from the catalyst to the good whose production it catalyses.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import autocatalytic_sets
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'tap_catalysis'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), observers=[observer], random_state=0).prepare().run(until=400)

# %%
# Hockey stick and the probability of a RAF
# -----------------------------------------
# The number of goods should stay near its start for most of the run and then shoot up within a short time. The
# largest RAF of the final production function network, found by the algorithm of Hordijk and Steel, should be absent
# or tiny at a low catalysis probability and present and close to the whole network at a high one, with about even
# chances near `pc = 0.0044` (Hordijk and Kauffman 2022, Fig. 2).

trajectory = simulation.trajectory_
goods = [len(trajectory.network(frame).nodes) for frame in range(len(trajectory.times))]
probabilities = [0.002, 0.003, 0.0044, 0.006, 0.008]
shares = []
for pc in probabilities:
    runs = [
        Simulation(create_system(SYSTEM, pc=pc), random_state=seed).prepare().run(until=400).trajectory_
        for seed in range(8)
    ]
    found = [autocatalytic_sets(run, frames=[-1]).fractions[0] for run in runs]
    shares.append(found)
fig, (growth, rafs) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(trajectory.times, goods)
growth.set_xlabel('time')
growth.set_ylabel('number of goods M')
growth.set_title('Hockey stick of TAP growth')
rafs.plot(probabilities, [np.mean(np.asarray(found) > 0.5) for found in shares], 'o-', label='Pr[large RAF]')
rafs.plot(probabilities, [np.mean(found) for found in shares], 's--', label='mean RAF share')
rafs.set_xlabel('catalysis probability pc')
rafs.set_ylabel('share of runs or of the network')
rafs.set_title('Autocatalytic sets of the final network')
rafs.legend()
fig.tight_layout()
plt.show()
