"""
# Kuramoto on hypergraphs

Oscillators coupled only through the triangles of a hypergraph feel `sin(theta_j + theta_k - 2 theta_i)` from the two
other members of every triangle. The incoherent state never loses stability, and above a coupling a synchronized state
coexists with it, so where the oscillators end depends on where they start.
"""

# %%
# Forty oscillators with Lorentzian frequencies of half width 0.5 sit on all the triangles of the complete hypergraph.
# The three-body coupling `K2 = 8` lies above the onset `8 x 0.5 = 4` of the synchronized branch. The run starts from
# phases spread over the circle and stays incoherent.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.synchronization import HypergraphKuramotoSystem

SYSTEM = 'hypergraph_kuramoto'

system = HypergraphKuramotoSystem(n_oscillators=40, K2=8.0, seed=0)
observables = HypergraphKuramotoSystem.observables
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='observables', interval=1)
trajectory = simulate(system, dt=0.5, observers=[observer], observables=observables)

# %%
# Hysteresis in the three-body coupling
# -------------------------------------
# A sweep of `K2` from incoherent phases keeps the order parameter near noise level, and a sweep from synchronized
# phases keeps it near one down to the onset of the branch, where it drops. The Ott-Antonsen branch
# `r^2 = (1 + sqrt(1 - 4 / K2)) / 2` is drawn for comparison.

couplings = np.linspace(1.0, 9.0, 5)
fig, ax = plt.subplots(figsize=(6, 4))
for label, phases in (('from incoherent phases', None), ('from synchronized phases', np.zeros(40))):
    order = []
    for coupling in couplings:
        system_system = HypergraphKuramotoSystem(n_oscillators=40, K2=coupling, phases=phases, seed=0)
        run = simulate(system_system, dt=0.5)
        order.append(np.mean(np.asarray(run.get_values('order'))[-20:]))
    ax.plot(couplings, order, 'o-', label=label)
branch = couplings[couplings >= 4.0]
ax.plot(branch, np.sqrt((1 + np.sqrt(1 - 4.0 / branch)) / 2), '--', color='black', label='Ott-Antonsen branch')
ax.set_xlabel('three-body coupling K2')
ax.set_ylabel('time-averaged order parameter r')
ax.set_title('Bistability of oscillators coupled by triangles')
ax.legend()
plt.show()
