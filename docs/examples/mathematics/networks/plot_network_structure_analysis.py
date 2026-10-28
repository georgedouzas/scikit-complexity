"""
# Centralities, assortativity, communities, and motifs

A stochastic block model of four planted groups splits into four communities: modularity maximization recovers the
planted blocks, and the edges packed inside the blocks close far more triangles than the degrees alone would.
"""

# %%
# The model is created by name from the catalog with its default network, a stochastic block model of four blocks of
# 100 nodes with mean in-block degree 16 and out-block degree 2. Every node is an agent that carries its degree,
# clustering, betweenness, and detected community, and the default blueprint draws the graph with the nodes colored by
# community.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import DistributionAnalysis, MotifAnalysis, NetworkAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'network_structure_analysis'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
model_system = create(MODEL, null_samples=0, random_state=0)
simulation = Simulation(
    model_system,
    dt=1,
    observers=[observer],
    observables=describe(MODEL).observables_of(model_system),
).run(until=1)
trajectory = simulation.trajectory_

# %%
# Degrees and motifs
# ------------------
# The degrees of a block model are close to Poisson around the mean degree of 18, so their survival function falls
# fast, without a heavy tail. The z-scores of open and closed triads against degree-preserving edge switches measure
# how far the triad counts lie from what the degrees alone give: the blocks pack edges together, so closed triangles
# should be over-represented with a large positive z-score.

network = NetworkAnalysis('degrees').check().prepare(trajectory).run()
graph, degrees = network.graphs_[-1], network.degrees_[-1]
ccdf = DistributionAnalysis('ccdf').check().prepare(degrees).run()
values, survival = ccdf.support_, ccdf.survival_
scores = MotifAnalysis('z-scores', samples=10, random_state=0).check().prepare(graph).run().z_scores_[0]
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.plot(values, survival, 'o-', markersize=3)
left.axvline(np.mean(degrees), color='black', linestyle=':', label='mean degree')
left.set_yscale('log')
left.set_xlabel('degree k')
left.set_ylabel('P(degree >= k)')
left.set_title(f"Degree distribution, modularity {trajectory.values('modularity')[-1]:.2f}")
left.legend()
right.bar(list(scores), list(scores.values()))
right.axhline(0, color='black', linewidth=0.8)
right.set_ylabel('z-score against edge switches')
right.set_title('Triad z-scores')
fig.tight_layout()
plt.show()
