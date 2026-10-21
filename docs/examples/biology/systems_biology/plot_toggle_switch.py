"""
# Genetic toggle switch

Two genes that repress each other form a bistable switch: whichever protein starts ahead silences the other gene and
stays on, and the cell remembers that choice.
"""

# %%
# The model is created by name from the catalog and started with the first protein ahead. The observer saves every
# frame with the default blueprint, whose primary view is the time series of the mRNA and protein counts.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'toggle_switch'

system = create(MODEL, proteins=(60.0, 0.0))
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# Two basins of attraction
# ------------------------
# Started from a grid of initial protein counts, every run should end with one gene on and the other off, and the
# diagonal where both start equal should separate the two outcomes.

grid = np.linspace(0, 120, 9)
fig, ax = plt.subplots(figsize=(6, 5))
for first in grid:
    for second in grid:
        trajectory = Simulation(create(MODEL, proteins=(first, second)), dt=0.5).run(until=100).trajectory_
        winner = 'tab:blue' if trajectory['gene_0.p'][-1] > trajectory['gene_1.p'][-1] else 'tab:orange'
        ax.plot(trajectory['gene_0.p'], trajectory['gene_1.p'], color=winner, linewidth=0.6, alpha=0.7)
ax.plot([0, 120], [0, 120], ':', color='black', label='equal start')
ax.set_xlabel('protein 0')
ax.set_ylabel('protein 1')
ax.set_title('Blue: gene 0 wins; orange: gene 1 wins')
ax.legend()
plt.show()
