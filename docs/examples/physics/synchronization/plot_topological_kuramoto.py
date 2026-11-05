"""
# Topological Kuramoto

Phases on the nodes and links of a clique complex couple through its incidence. The link phases feel the sines of their
projections on the nodes and on the triangles, and the two layers scale each other's coupling by their order parameters.
"""

# %%
# A run at a coupling above the transition shows the order parameter of the projections of the link phases rise.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'topological_kuramoto'

system = create(MODEL, coupling=8.0)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
Simulation(system, dt=0.5, observers=[observer], observables=observables).prepare().run(until=20)

# %%
# Synchronization against the coupling
# ------------------------------------
# The order parameter of the link phases at the end of a run grows from its noise level at zero coupling.

couplings = np.linspace(0.0, 8.0, 5)
orders = []
for coupling in couplings:
    model_system = create(MODEL, coupling=coupling)
    run = (
        Simulation(model_system, dt=0.5, observables=describe(MODEL).observables_of(model_system))
        .prepare()
        .run(until=20)
    )
    orders.append(np.mean(run.trajectory_.values('link_order')[-10:]))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(couplings, orders, 'o-')
ax.set_xlabel('coupling K')
ax.set_ylabel('order parameter of the link phases')
ax.set_title('Synchronization of the phases on links')
plt.show()
