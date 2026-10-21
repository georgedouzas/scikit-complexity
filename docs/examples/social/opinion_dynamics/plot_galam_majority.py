"""
# Galam majority

People meet in random groups of three, and every group adopts the opinion of its majority. Each round amplifies the
larger camp, so a minority of 40% shrinks round after round until everyone holds the majority view.
"""

# %%
# Three hundred people hold opinion 1 with probability 0.4, 34% of them in this draw. The default blueprint shows the
# share of opinion 1, the mean of the opinions, falling to zero within four rounds, with the spread of opinions
# vanishing as consensus forms.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'galam_majority'

system = create(MODEL, share=0.4, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=10)

# %%
# One round of discussions
# ------------------------
# In groups of three, a camp holding the share `p` keeps the groups where it has two or three members, so after one
# round its share should be `p^3 + 3 p^2 (1 - p)`, a map with the unstable fixed point 1/2 that drives every other
# share to 0 or 1.


def after_one_round(share, seed):
    """Return the share of opinion 1 after one round of discussions in a large population."""
    run = Simulation(create(MODEL, n_agents=3000, share=share, random_state=seed), random_state=seed).run(until=1)
    return run.trajectory_.agents('citizens', 1).values['opinion'].mean()


shares = np.linspace(0.05, 0.95, 10)
simulated = [np.mean([after_one_round(share, seed) for seed in range(2)]) for share in shares]
grid = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(grid, grid**3 + 3 * grid**2 * (1 - grid), color='black', label='p^3 + 3 p^2 (1 - p)')
ax.plot(grid, grid, color='grey', linestyle=':', label='no change')
ax.plot(shares, simulated, 'o', label='simulated, 3000 people')
ax.set_xlabel('share of opinion 1 before a round')
ax.set_ylabel('share after the round')
ax.set_title('Majority rule amplifies the larger camp')
ax.legend()
plt.show()
