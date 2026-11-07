"""
# Self-organized branching process

Avalanches spread down a binary tree: every active site wakes two sites below it with the probability `p`. Energy enters
with every avalanche and leaves through the sites that reach the bottom, and this balance tunes `p` to the critical
value `1 / 2` without any adjustment from outside, a mean-field picture of self-organized criticality.
"""

# %%
# The tree has ten generations, and the branching probability starts at 0.3. One step is one generation of an
# avalanche. The default blueprint shows `p`, the active sites, and the generation.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'self_organized_branching'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=50)
simulation = create_simulation(SYSTEM).set_params(observers=[observer], random_state=0).prepare().run(until=30000)

# %%
# Approach to the critical point
# ------------------------------
# Started below or above `1 / 2`, the branching probability should drift to it and then fluctuate around it.

fig, ax = plt.subplots(figsize=(6, 4))
for start in (0.3, 0.7):
    run = create_simulation(SYSTEM, p=start).set_params(random_state=0).prepare().run(until=30000)
    ax.plot(np.asarray(run.trajectory_.times), np.ravel(run.trajectory_.values('tree.p')), label=f'p(0) = {start}')
ax.axhline(0.5, linestyle='--', color='black')
ax.set_xlabel('generations')
ax.set_ylabel('branching probability p')
ax.set_title('Self-organization to p = 1/2')
ax.legend()
plt.show()
