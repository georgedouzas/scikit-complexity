"""
# Adaptive Kuramoto-Sakaguchi oscillators with frequency clusters

Identical phase oscillators are coupled through weights that adapt slowly to their phase differences. Instead of one
synchronized group, the population splits into frequency clusters, groups strongly coupled inside and weakly coupled
across, each moving at its own frequency.
"""

# %%
# Fifty oscillators start at random phases with random weights, with the phase lags `alpha = 0.3 π` and
# `beta = 0.23 π` and the adaptation rate `epsilon = 0.01` of Fig. 12 of Berner et al. The default blueprint shows the
# phases on the nodes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.synchronization import AdaptiveKuramotoSystem

SYSTEM = 'adaptive_kuramoto'

system = AdaptiveKuramotoSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=20)
trajectory = simulate(system, observers=[observer])

# %%
# Clusters and their weights
# --------------------------
# Sorting the oscillators by their mean frequency over the last 200 time units should show a few plateaus, the
# clusters, and the matrix of weights in that order should show strong blocks on the diagonal, the clusters, with weak
# weights between them.

theta = np.asarray(trajectory.values('nodes.theta'))
frequencies = (theta[-1] - theta[-201]) / 200
order = np.argsort(frequencies)
sources, targets = system.prepare().system_.elements[0].space.prepare().edges_[None]
weights = np.zeros((len(order), len(order)))
weights[targets, sources] = np.asarray(trajectory.values('links.kappa'))[-1]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(frequencies[order], 'o')
axes[0].set_xlabel('oscillator, by frequency')
axes[0].set_ylabel('mean frequency')
axes[1].imshow(weights[np.ix_(order, order)], cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_xlabel('oscillator j')
axes[1].set_ylabel('oscillator i')
axes[1].set_title('weights kappa_ij')
plt.show()
