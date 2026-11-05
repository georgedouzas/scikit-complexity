"""
# Gauge network agents

Agents on a ring relax towards the phases of their neighbours, transported through a connection on every edge. A flat
connection lets them agree. A flux of pi around the ring frustrates the loop, so a residual energy and a twist remain.
"""

# %%
# A run with a flux of pi shows the phases settling into a twist around the ring.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'gauge_network_agents'

system = create(MODEL, flux=np.pi)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
Simulation(system, dt=0.5, observers=[observer], observables=observables).prepare().run(until=40)

# %%
# Frustration blocks agreement
# ----------------------------
# The energy of the flat ring decays to zero, and that of the frustrated ring stops at `12 (1 - cos(pi / 12))`.

fig, ax = plt.subplots(figsize=(6, 4))
for label, flux in (('flat connection', 0.0), ('flux of pi', np.pi)):
    model_system = create(MODEL, flux=flux)
    run = (
        Simulation(model_system, dt=0.5, observables=describe(MODEL).observables_of(model_system))
        .prepare()
        .run(until=40)
    )
    ax.semilogy(run.trajectory_.times, run.trajectory_.values('energy') + 1e-16, label=label)
ax.set_xlabel('time')
ax.set_ylabel('energy of the transported differences')
ax.legend()
plt.show()
