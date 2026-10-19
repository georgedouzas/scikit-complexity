"""
# Cell-cycle chemotherapy

A killing agent that kills 60% of the dividing cells, with a blocking agent that slows the S phase, makes every
compartment of the cell cycle decay after an initial redistribution of the cells.
"""

# %%
# The model is created by name from the catalog with the three compartments of Table 3.8, a killing agent `u1 = 0.6`,
# and a blocking agent `u2 = 0.3`. The observer saves every frame with the default blueprint, whose primary view is the
# time series of the three compartments.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cell_cycle_chemotherapy'

system = create(MODEL, u1=0.6, u2=0.3)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# The kill threshold
# ------------------
# Every division gives `2 (1 - u1)` cells, so the tumour grows for a kill below one half and shrinks above it. The
# late growth rate of the total should equal the dominant eigenvalue of the linear cycle and change sign at
# `u1 = 1/2`, whatever the blocking.

l1, l2, l3, u2 = 0.197, 0.395, 0.107, 0.3
kills = np.linspace(0.0, 0.8, 9)
rates, eigenvalues = [], []
for u1 in kills:
    trajectory = Simulation(create(MODEL, u1=u1, u2=u2), dt=2.0).run(until=300).trajectory_
    total = np.log(trajectory['tumour.A1'] + trajectory['tumour.A2'] + trajectory['tumour.A3'])
    late = trajectory.times >= 200
    rates.append(np.polyfit(trajectory.times[late], total[late], 1)[0])
    matrix = [[-l1, 0, 2 * (1 - u1) * l3], [l1, -(1 - u2) * l2, 0], [0, (1 - u2) * l2, -l3]]
    eigenvalues.append(np.linalg.eigvals(matrix).real.max())
fig, (compartments, threshold) = plt.subplots(1, 2, figsize=(10, 4))
for name in ('A1', 'A2', 'A3'):
    compartments.plot(simulation.trajectory_.times, simulation.trajectory_[f'tumour.{name}'], label=name)
compartments.set_xlabel('time')
compartments.set_ylabel('cells')
compartments.set_title('Under u1 = 0.6 every compartment decays')
compartments.legend()
threshold.plot(kills, rates, 'o', label='simulated')
threshold.plot(kills, eigenvalues, '-', color='black', label='dominant eigenvalue')
threshold.axhline(0, color='grey', linestyle=':')
threshold.axvline(0.5, color='grey', linestyle='--', label='u1 = 1/2')
threshold.set_xlabel('fraction of dividing cells killed u1')
threshold.set_ylabel('growth rate')
threshold.set_title('The tumour shrinks above a kill of one half')
threshold.legend()
fig.tight_layout()
plt.show()
