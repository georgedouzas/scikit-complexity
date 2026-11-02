"""
# Nonlinear consensus on a temporal hypergraph

On every active triangle of a hypergraph a node moves towards the two other members, with a pull that decays with the
distance between them. Triangles switch on and off on their own schedules. The opinions contract, and the reinforcement
makes the mean opinion drift towards the pairs that agree.
"""

# %%
# Thirty nodes hold opinions drawn between -1 and 1, and every triple of nodes is a triangle with probability 0.05. The
# default blueprint shows the mean opinion and the spread.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'hypergraph_consensus'

system = create(MODEL)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', primary='observables', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer], observables=observables).check().prepare().run(until=10.0)

# %%
# The mean opinion
# ----------------
# Without reinforcement the pulls of a triangle on its three members sum to zero, so the mean opinion should stay fixed
# whatever the schedules. A negative reinforcement breaks that symmetry and the mean opinion should move away from its
# initial value, while the opinions still contract.

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for reinforcement in (0.0, -1.0):
    run = Simulation(create(MODEL, reinforcement=reinforcement), dt=0.1).check().prepare().run(until=10.0)
    opinions = np.asarray(run.trajectory_.values('nodes.x'))
    opinions = opinions.reshape(len(opinions), -1)
    label = f'reinforcement {reinforcement}'
    axes[0].plot(opinions.mean(axis=1), label=label)
    axes[1].plot(opinions.std(axis=1), label=label)
axes[0].set_xlabel('step')
axes[0].set_ylabel('mean opinion')
axes[1].set_xlabel('step')
axes[1].set_ylabel('standard deviation of the opinions')
axes[0].legend()
fig.suptitle('Reinforcement moves the mean, and the opinions contract')
fig.tight_layout()
plt.show()
