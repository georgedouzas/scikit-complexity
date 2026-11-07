"""
# Hypercycles

Self-replicating molecules that catalyse each other in a ring keep every member alive, although some replicate faster
than others. With five or more molecules the concentrations chase each other round the ring: each molecule in turn
dominates the soup, boosts the next one, and gives way to it.
"""

# %%
# The system is built from its class as a ring of six molecules whose own growth rates differ only slightly, `alpha =
# 0.01`, so that catalysis dominates. The observer saves every frame with the default blueprint, whose primary view is
# the time series of the concentrations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import HypercyclesSystem

SYSTEM = 'hypercycles'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(
    HypercyclesSystem(molecules=6, alpha=0.01, topology='cycle'),
    dt=1.0,
    observers=[observer],
    until=290,
)

# %%
# Cycle against complete network
# ------------------------------
# In the ring every molecule should keep coming back, so on a logarithmic scale each concentration should rise and
# fall periodically without dying out. When every molecule catalyses every other, the fastest replicators should take
# over instead: about `sqrt(2 omega / alpha) ≈ 4.5` of them survive for `alpha = 0.1` and the slowest die out.

ring = trajectory
complete = simulate(HypercyclesSystem(molecules=10, topology='complete'), until=200).values('soup.x')
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
left.semilogy(ring.times, ring.values('soup.x'))
left.set_ylim(1e-5, 1)
left.set_xlabel('time')
left.set_ylabel('concentration (log scale)')
left.set_title('Ring of six: the molecules take turns')
right.bar(np.arange(1, 11), complete[-1])
right.set_xlabel('molecule (faster replicators to the right)')
right.set_ylabel('final concentration')
right.set_title(f'Complete network of ten: {np.sum(complete[-1] > 1e-3)} survive')
fig.tight_layout()
plt.show()
