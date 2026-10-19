"""
# Prostate cancer with an androgen cell quota

During each therapy period the serum androgen falls, the cells use up their androgen quota and shrink, and the PSA
falls. Off therapy the quotas refill and the tumour and the PSA regrow.
"""

# %%
# The model is created by name from the catalog: 15 billion androgen-dependent cells under therapy for 250 days every
# 500 days. The observer saves every second frame with the default blueprint, whose primary view is the time series of
# the cells, the quotas, the PSA, and the androgen.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'prostate_cell_quota'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=5.0, observers=[observer]).run(until=1500)

# %%
# Quotas set the growth
# ---------------------
# The dependent cells grow at `rm (1 - q1 / Q1)`, so they shrink whenever the quota falls below
# `rm q1 / (rm - d1) = 0.75` nM and grow above it. The growth rate should change sign with the therapy, and the PSA
# should follow the cells.

trajectory = simulation.trajectory_
times = trajectory.times
fig, (quota, cells) = plt.subplots(1, 2, figsize=(10, 4))
quota.plot(times, trajectory['tumour.Q1'], label='dependent cells Q1')
quota.plot(times, trajectory['tumour.Q2'], label='independent cells Q2')
quota.axhline(0.025 * 0.3 / (0.025 - 0.015), color='black', linestyle='--', label='no net growth of AD')
quota.set_xlabel('time (days)')
quota.set_ylabel('cell quota (nM)')
quota.set_title('The quotas follow the androgen')
quota.legend()
cells.plot(times, trajectory['tumour.AD'], label='dependent cells (billions)')
cells.plot(times, trajectory['tumour.P'], label='serum PSA (ng/mL)')
cells.set_xlabel('time (days)')
cells.set_title('Cells and PSA fall during therapy')
cells.legend()
fig.tight_layout()
plt.show()
