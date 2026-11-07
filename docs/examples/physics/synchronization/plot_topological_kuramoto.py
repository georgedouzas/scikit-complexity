"""
# Topological Kuramoto

Phases on the nodes and links of a clique complex couple through its incidence. The link phases feel the sines of their
projections on the nodes and on the triangles, and the two layers scale each other's coupling by their order parameters.
"""

# %%
# A run at a coupling above the transition shows the order parameter of the projections of the link phases rise.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.synchronization import TopologicalKuramotoSystem

SYSTEM = 'topological_kuramoto'

system = TopologicalKuramotoSystem(coupling=8.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulate(system, until=20, dt=0.5, observers=[observer])

# %%
# Synchronization against the coupling
# ------------------------------------
# The order parameter of the link phases at the end of a run grows from its noise level at zero coupling.

couplings = np.linspace(0.0, 8.0, 5)
orders = []
for coupling in couplings:
    system_system = TopologicalKuramotoSystem(coupling=coupling)
    run = simulate(system_system, until=20, dt=0.5)
    orders.append(np.mean(run.values('link_order')[-10:]))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(couplings, orders, 'o-')
ax.set_xlabel('coupling K')
ax.set_ylabel('order parameter of the link phases')
ax.set_title('Synchronization of the phases on links')
plt.show()

# %%
# Nodes-Links-Triangles coupling
# ------------------------------
# In the Nodes-Links-Triangles system every coupling is scaled by the order of another projection, so the node, link,
# and triangle phases should all stay incoherent at weak coupling and synchronize together at a stronger one, while the
# nodes of the Nodes-Links system already synchronize at weak coupling.

weak = [0.25, 0.5, 1.0, 2.0]
fig, ax = plt.subplots(figsize=(6, 4))
for model in ('NL', 'NLT'):
    orders = []
    for coupling in weak:
        system_system = TopologicalKuramotoSystem(coupling=coupling, model=model)
        run = simulate(system_system, until=20, dt=0.5)
        orders.append(np.mean(run.values('node_order')[-10:]))
    ax.plot(weak, orders, 'o-', label=model)
ax.set_xlabel('coupling K')
ax.set_ylabel('order parameter of the node phases')
ax.set_title('Nodes-Links and Nodes-Links-Triangles systems')
ax.legend()
plt.show()
