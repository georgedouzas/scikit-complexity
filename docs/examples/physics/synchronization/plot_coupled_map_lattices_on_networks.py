"""
# Coupled map lattices on networks

Chaotic logistic maps on the nodes of a random network, each pulled towards the mean of its neighbours, fall onto one
chaotic orbit when the coupling is strong: the nodes take one color that keeps changing from step to step.
"""

# %%
# A hundred maps `f(x) = 4 x (1 - x)` sit on an Erdős-Rényi network of mean degree 20 with the coupling
# `epsilon = 0.9`, and start from random states. The blueprint shows the network colored by state, which becomes
# uniform once the maps synchronize, with the variance of the states over the nodes below it.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create, describe

MODEL = 'coupled_map_lattices_on_networks'

system = create(MODEL, n_nodes=100, degree=20, epsilon=0.9, seed=0)
observables = describe(MODEL).observables_of(system)
blueprint = default_blueprint(system, primary='maps/x', observables=observables)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], observables=observables).run(until=60)

# %%
# Synchronization window
# ----------------------
# A perturbation of the synchronized orbit along an eigenvector of the normalized Laplacian with eigenvalue `lambda_k`
# grows by `|1 - epsilon lambda_k| e^h` per step, with the Lyapunov exponent `h = ln 2` of the map. The maps should
# synchronize exactly where `|1 - epsilon lambda_k| < 1 / 2` for every nonzero eigenvalue, the window between the dashed
# lines, and the late variance of the states should vanish inside it only.

space = system.components[0].space
eigenvalues = np.sort(nx.normalized_laplacian_spectrum(space.graph))
low, high = 0.5 / eigenvalues[1], 1.5 / eigenvalues[-1]
couplings = np.linspace(0.05, 1.0, 20)
variances = []
for coupling in couplings:
    model_system = create(MODEL, n_nodes=100, degree=20, epsilon=coupling, seed=0)
    run = Simulation(model_system, dt=1.0, observables=describe(MODEL).observables_of(model_system)).run(until=200)
    variances.append(np.mean(np.asarray(run.trajectory_['spread'])[-20:]))
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(couplings, np.maximum(variances, 1e-32), 'o-', label='late variance of the states')
for bound in (low, high):
    ax.axvline(bound, linestyle='--', color='black')
ax.set_xlabel('coupling epsilon')
ax.set_ylabel('variance over the nodes')
ax.set_title('Synchronization window of logistic maps on a network')
ax.legend()
plt.show()
