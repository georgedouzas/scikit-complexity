"""
# Core and periphery vasculature

Under a weekly anti-angiogenic infusion the starved tumour recruits few vessels, so its radius stays near 13 mm and its
live cells grow only to about `3 10^4` in two months, against about `10^11` without treatment.
"""

# %%
# The model is created by name from the catalog with its default schedule, a one-day infusion of 5 mg/kg/day every
# week, from the book's initial state of a sparse tumour of radius 13 mm without vessels. The system is stiff, so it
# runs with LSODA. The observer saves every frame with the default blueprint, whose primary view is the time series of
# the radius, the cells, the vascular volumes, and the inhibitor.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'vascular_core_periphery'

solver = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-8)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, solver=solver, observers=[observer]).run(until=60)

# %%
# Treated and untreated tumours
# -----------------------------
# Untreated, the vessels the starved cells recruit should feed the periphery, the live cells should grow by orders of
# magnitude, and the radius should expand once the periphery is dense. Under the inhibitor the vessels and the live
# cells should stay orders of magnitude lower and the radius close to its initial 13 mm.

treated = simulation.trajectory_
untreated = Simulation(create(MODEL, dose=0.0), dt=0.5, solver=solver).run(until=60).trajectory_
fig, (cells, vessels) = plt.subplots(1, 2, figsize=(10, 4))
for trajectory, name in ((untreated, 'untreated'), (treated, 'treated')):
    cells.semilogy(trajectory.times, trajectory['tumour.Ap'] + trajectory['tumour.Ac'], label=f'live cells, {name}')
    cells.semilogy(trajectory.times, trajectory['tumour.An'].clip(1e-3), linestyle=':', label=f'necrotic, {name}')
    vessels.semilogy(trajectory.times, trajectory['tumour.Vbp'].clip(1e-6), label=f'periphery, {name}')
    vessels.semilogy(trajectory.times, trajectory['tumour.Vbc'].clip(1e-6), linestyle=':', label=f'core, {name}')
cells.set_xlabel('time (days)')
cells.set_ylabel('cells')
cells.set_title('The inhibitor holds the cells down')
cells.legend()
vessels.set_xlabel('time (days)')
vessels.set_ylabel('vascular volume (mm³)')
vessels.set_title('The inhibitor prunes the vasculature')
vessels.legend()
fig.tight_layout()
plt.show()
