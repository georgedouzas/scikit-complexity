"""
# Watts threshold cascades

Every node adopts once the share of its adopting neighbours exceeds its threshold. A single seed either fizzles out
at once or sets off a global cascade that sweeps almost the whole network, and which one happens depends on chance.
"""

# %%
# The model is created by name from the catalog with the default random network, the threshold 0.18 for every node,
# and one random seed, and runs for 30 steps. The default blueprint draws the network, where the adopters are yellow.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'watts_cascades'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=1, observers=[observer]).run(until=30)

# %%
# The cascade window
# ------------------
# Cascades are all or nothing: the final share of adopters should be either near zero or near one. As the threshold
# grows, fewer nodes are vulnerable enough to pass the cascade on from a single neighbour, so the probability of a
# global cascade should fall to zero beyond a critical threshold.

thresholds = np.linspace(0.1, 0.34, 13)
probabilities, shares = [], []
for threshold in thresholds:
    finals = []
    for seed in range(30):
        run = Simulation(create(MODEL, threshold=float(threshold), random_state=seed), dt=1).run(until=30).trajectory_
        finals.append(run.agents('nodes', len(run.times) - 1).values['active'].mean())
    shares.extend(finals)
    probabilities.append(np.mean(np.array(finals) > 0.5))
fig, (window, outcomes) = plt.subplots(1, 2, figsize=(10, 4))
window.plot(thresholds, probabilities, 'o-')
window.set_xlabel('threshold')
window.set_ylabel('probability of a global cascade')
window.set_title('Global cascades stop beyond a critical threshold')
outcomes.hist(shares, bins=20)
outcomes.set_xlabel('final share of adopters')
outcomes.set_ylabel('runs')
outcomes.set_title('Cascades are all or nothing')
fig.tight_layout()
plt.show()
