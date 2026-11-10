"""
# Network superposition model

The network is made of many Erdős-Rényi subgraphs, each with its own linking probability drawn from a distribution
`Q(p)`. The degrees of every subgraph are Poisson, and the network mixes them, so a power law of the linking
probabilities gives a power law of the degrees.
"""

# %%
# Twenty subgraphs of a hundred nodes draw their linking probabilities from `Q(p) ~ p^-2` on `[0.005, 0.1]`.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import NetworkSuperpositionSystem

SYSTEM = 'network_superposition'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(NetworkSuperpositionSystem(), observers=[observer])

# %%
# Degree distribution
# -------------------
# The degrees should follow Eq. 4.34 of Thurner et al., `P(k) = ∫ dp P(k | N p) Q(p)`, the mixture of the Poisson
# degrees of the subgraphs, here for 1000 subgraphs.

low, high, exponent, n_nodes = 0.005, 0.1, 2.0, 100
degrees = np.asarray(
    NetworkSuperpositionSystem(n_subgraphs=1000).prepare().system_.elements[0].values['degree'],
    dtype=int,
)
norm = (low ** (1 - exponent) - high ** (1 - exponent)) / (exponent - 1)
support = np.arange(degrees.max() + 1)
mixture = [quad(lambda p, k=k: poisson.pmf(k, (n_nodes - 1) * p) * p**-exponent / norm, low, high)[0] for k in support]
fig, axis = plt.subplots(figsize=(6, 4))
counts = np.bincount(degrees) / degrees.size
axis.semilogy(support, counts, 'o', label='simulated')
axis.semilogy(support, mixture, 'k-', label='Eq. 4.34')
axis.set_xlabel('degree k')
axis.set_ylabel('share of nodes')
axis.set_title('Superposed Poisson degrees')
axis.legend()
fig.tight_layout()
plt.show()
