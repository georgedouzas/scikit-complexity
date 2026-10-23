"""
# Kuramoto on networks

Oscillators on the nodes of a small-world network pull their neighbours along the edges. With strong enough coupling the
phases lock across the whole network, and the order parameter `r` climbs from noise level to near one.
"""

# %%
# Two hundred oscillators sit on a Watts-Strogatz ring of mean degree 6 with a tenth of its edges rewired, and their
# frequencies have width `sigma = 0.5`. The coupling `K = 3` per mean degree is well above the onset. The blueprint puts
# the order parameter first, with the network colored by phase in a second view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create, describe

MODEL = 'kuramoto_on_networks'

system = create(MODEL, n_oscillators=200, K=3.0, sigma=0.5, seed=0)
observables = describe(MODEL).observables_of(system)
blueprint = default_blueprint(system, primary='observables', observables=observables)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=0.2, observers=[observer], observables=observables).run(until=30)

# %%
# Onset of synchronization on two networks
# ----------------------------------------
# With the coupling divided by the mean degree, a dense random network acts much like all-to-all coupling, whose onset
# for normal frequencies is `K_c = 2 sigma sqrt(2 / pi)`, about 0.8 here. The time-averaged order parameter should stay
# near `1 / sqrt(N)` below the onset and rise above it, a little later on the clustered small-world ring than on the
# random network.

couplings = np.linspace(0.0, 3.0, 13)
fig, ax = plt.subplots(figsize=(6, 4))
for network in ('erdos_renyi', 'watts_strogatz'):
    order = []
    for coupling in couplings:
        model_system = create(MODEL, n_oscillators=200, network=network, K=coupling, sigma=0.5, seed=1)
        run = Simulation(model_system, dt=0.5, observables=describe(MODEL).observables_of(model_system)).run(until=40)
        order.append(np.mean(np.asarray(run.trajectory_['order'])[-20:]))
    ax.plot(couplings, order, 'o-', label=network.replace('_', '-'))
ax.axvline(2 * 0.5 * np.sqrt(2 / np.pi), linestyle='--', color='black', label='all-to-all K_c')
ax.axhline(1 / np.sqrt(200), linestyle=':', color='grey', label='1 / sqrt(N)')
ax.set_xlabel('coupling K')
ax.set_ylabel('time-averaged order parameter r')
ax.set_title('Synchronization of 200 oscillators on networks')
ax.legend()
plt.show()
