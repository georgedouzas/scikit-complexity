"""
# Chemoimmunotherapy

The tumour cells regrow between three weekly doses of chemotherapy, IL-2, and tumour-infiltrating lymphocytes and are
gone after the third, while the NK cells expand and the circulating lymphocytes recover from the drug.
"""

# %%
# The system is built from its class with the mouse of de Pillis et al.: a million tumour cells, few NK and CD8 T
# cells, and circulating lymphocytes at their balance. All three drugs are given for one day every week from day 7.
# The observer saves every frame with the default blueprint, whose primary view is the time series of the cells and
# the drugs.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import ChemoimmunotherapySystem

SYSTEM = 'chemoimmunotherapy'
SOLVER = ScipyODESolver(method='LSODA', rtol=1e-6, atol=1e-6)

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ChemoimmunotherapySystem(), dt=0.25, solver=SOLVER, observers=[observer])

# %%
# Each therapy alone and both together
# ------------------------------------
# Untreated, the tumour settles near 3.4e7 cells. Chemotherapy alone and immunotherapy alone only delay it, and it
# returns to that level. The combination should drive the tumour below one cell.

arms = {
    'untreated': ChemoimmunotherapySystem(u_til=0.0, u_il2=0.0, u_chemo=0.0).prepare().system_,
    'chemotherapy alone': ChemoimmunotherapySystem(u_til=0.0, u_il2=0.0).prepare().system_,
    'immunotherapy alone': ChemoimmunotherapySystem(u_chemo=0.0).prepare().system_,
}
fig, (tumour, tcells) = plt.subplots(1, 2, figsize=(10, 4))
for name, arm in arms.items():
    trajectory = simulate(arm, dt=0.25, solver=SOLVER, until=60)
    tumour.semilogy(trajectory.times, trajectory.get_values('tumour.A').clip(1e-2), label=name)
    tcells.semilogy(trajectory.times, trajectory.get_values('tumour.E').clip(1e-2), label=name)
combined = trajectory
tumour.semilogy(combined.times, combined.get_values('tumour.A').clip(1e-2), label='combination', color='black')
tcells.semilogy(combined.times, combined.get_values('tumour.E').clip(1e-2), label='combination', color='black')
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
