"""
# Biphasic growth of an off-lattice tumour

Cell centres in a square are joined by springs along the edges of their Delaunay triangulation. Cancer cells at the
edge of the tumour divide once old enough and invade healthy neighbours, and the age a cell needs to divide drops at a
switch day, so the tumour grows slowly first and fast after the switch.
"""

# %%
# The tissue is a hexagonal layout of healthy cells in a square of 12 cell diameters, with a small cluster of cancer
# cells at its centre. Cancer cells need 20 hours to divide before day 10 and 5 hours after it. Time runs in hours and
# the run lasts 20 days. The observer saves a frame every day with the default blueprint.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.oncology import OffLatticeTumourGrowthSystem

SYSTEM = 'off_lattice_tumour_growth'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=24)
trajectory = simulate(OffLatticeTumourGrowthSystem(), until=480, dt=24, observers=[observer], random_state=0)

# %%
# Tumour area and the tissue at the end
# -------------------------------------
# The tumour area should rise slowly before day 10 and steeply after it, until the tumour fills the square. The cells
# at the end show the tumour in red among the healthy cells, spaced by their springs.

days = np.asarray(trajectory.times) / 24
locations = trajectory.get_agent_locations('cells')[-1]
cancer = trajectory.get_agents('cells', len(trajectory.times) - 1).values['cancer']
fig, (area, tissue) = plt.subplots(1, 2, figsize=(10, 4.5))
area.plot(days, trajectory.get_values('tumour_area'), color='tab:red')
area.axvline(10, color='black', linestyle='--', label='switch')
area.set_xlabel('time (days)')
area.set_ylabel('tumour area (cell diameters squared)')
area.set_title('Biphasic growth')
area.legend()
tissue.scatter(*locations.T, c=np.where(cancer > 0, 'tab:red', 'tab:blue'), s=40)
tissue.set_aspect('equal')
tissue.set_title('Cells at day 20')
fig.tight_layout()
plt.show()

# %%
# Calibrating the second phase by ABC-SMC
# ---------------------------------------
# The tumour areas of a smaller tissue every two days over 12 days, with the switch at day 4, are the data of one run
# at `g_age2 = 5` hours. ABC-SMC compares the areas of every simulation with them, with a uniform prior on `g_age2`
# between 2 and 24 hours, as Wang et al. use, and a small budget of simulations. The weighted posterior should favour
# short ages.

small = {'side': 8.0, 'tau': 4.0}
truth = simulate(OffLatticeTumourGrowthSystem(**small, g_age2=5.0), until=288, dt=24, random_state=1)
times = np.arange(0.0, 289.0, 48.0)
data = {'tumour_area': (times, np.interp(times, truth.times, truth.get_values('tumour_area')))}
posterior = calibrate(
    OffLatticeTumourGrowthSystem(**small),
    data,
    {'g_age2': (2.0, 24.0)},
    method='abc_smc',
    n_samples=30,
    budget=150,
    dt=24,
    random_state=0,
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(posterior.samples['g_age2'], bins=15, weights=posterior.weights, color='tab:red', alpha=0.7, label='posterior')
ax.axvline(5.0, color='black', linestyle='--', label='true age')
ax.axvline(posterior.estimates['g_age2'], color='tab:blue', label=f'median {posterior.estimates["g_age2"]:.1f}')
ax.set_xlabel('age to divide after the switch (hours)')
ax.set_ylabel('weight')
ax.set_title('ABC-SMC on the tumour areas')
ax.legend()
plt.show()
