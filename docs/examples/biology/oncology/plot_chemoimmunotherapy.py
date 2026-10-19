"""
# Chemoimmunotherapy

The tumour cells regrow between three weekly doses of chemotherapy, IL-2, and tumour-infiltrating lymphocytes and are
gone after the third, while the NK cells expand and the circulating lymphocytes recover from the drug.
"""

# %%
# The model is created by name from the catalog with the mouse of de Pillis et al.: a million tumour cells, few NK and
# CD8 T cells, and circulating lymphocytes at their balance. All three drugs are given for one day every week from day
# 7. The observer saves every frame with the default blueprint, whose primary view is the time series of the cells and
# the drugs.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'chemoimmunotherapy'
SOLVER = ScipyODE(method='LSODA', rtol=1e-6, atol=1e-6)

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.25, solver=SOLVER, observers=[observer]).run(until=60)

# %%
# Each therapy alone and both together
# ------------------------------------
# Untreated, the tumour settles near 3.4e7 cells. Chemotherapy alone and immunotherapy alone only delay it, and it
# returns to that level. The combination should drive the tumour below one cell.

arms = {
    'untreated': create(MODEL, u_til=0.0, u_il2=0.0, u_chemo=0.0),
    'chemotherapy alone': create(MODEL, u_til=0.0, u_il2=0.0),
    'immunotherapy alone': create(MODEL, u_chemo=0.0),
}
fig, (tumour, tcells) = plt.subplots(1, 2, figsize=(10, 4))
for name, arm in arms.items():
    trajectory = Simulation(arm, dt=0.25, solver=SOLVER).run(until=60).trajectory_
    tumour.semilogy(trajectory.times, trajectory['tumour.A'].clip(1e-2), label=name)
    tcells.semilogy(trajectory.times, trajectory['tumour.E'].clip(1e-2), label=name)
combined = simulation.trajectory_
tumour.semilogy(combined.times, combined['tumour.A'].clip(1e-2), label='combination', color='black')
tcells.semilogy(combined.times, combined['tumour.E'].clip(1e-2), label='combination', color='black')
tumour.axhline(1.0, color='grey', linestyle=':', label='one cell')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells')
tumour.set_title('Only the combination clears the tumour')
tumour.legend()
tcells.set_xlabel('time (days)')
tcells.set_ylabel('CD8 T cells')
tcells.set_title('The injected T cells meet a smaller tumour')
tcells.legend()
fig.tight_layout()
plt.show()
