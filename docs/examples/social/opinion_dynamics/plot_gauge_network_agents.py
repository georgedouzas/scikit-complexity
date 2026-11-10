"""
# Gauge network agents

Agents on a ring relax towards the phases of their neighbours, transported through a connection on every edge. A flat
connection lets them agree. A flux of pi around the ring frustrates the loop, so a residual energy and a twist remain.
"""

# %%
# A run with a flux of pi shows the phases settling into a twist around the ring.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.opinion_dynamics import GaugeNetworkAgentsSystem

SYSTEM = 'gauge_network_agents'

system = GaugeNetworkAgentsSystem(flux=np.pi)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulate(system, until=40, observers=[observer])

# %%
# Frustration blocks agreement
# ----------------------------
# The energy of the flat ring decays to zero, and that of the frustrated ring stops at `12 (1 - cos(pi / 12))`.

fig, ax = plt.subplots(figsize=(6, 4))
for label, flux in (('flat connection', 0.0), ('flux of pi', np.pi)):
    system_system = GaugeNetworkAgentsSystem(flux=flux)
    run = simulate(system_system, until=40)
    ax.semilogy(run.times, run.get_values('energy') + 1e-16, label=label)
ax.set_xlabel('time')
ax.set_ylabel('energy of the transported differences')
ax.legend()
plt.show()
