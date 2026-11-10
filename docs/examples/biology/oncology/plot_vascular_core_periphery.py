"""
# Core and periphery vasculature

Under a weekly anti-angiogenic infusion the starved tumour recruits few vessels, so its radius stays near 13 mm and its
live cells grow only to about `3 10^4` in two months, against about `10^11` without treatment.
"""

# %%
# The example builds `VascularCorePeripherySystem` with its default schedule, a one-day infusion of 5 mg/kg/day every
# week, from the initial state of the book of Padmanabhan, Meskin, and Al Moustafa (2021), a sparse tumour of radius 13
# mm without vessels. The system is stiff, so it runs with LSODA. The observer saves every frame with the default
# blueprint, whose primary view is the time series of the radius, the cells, the vascular volumes, and the inhibitor.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import VascularCorePeripherySystem

SYSTEM = 'vascular_core_periphery'

solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-8)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(VascularCorePeripherySystem(), dt=0.5, solver=solver, observers=[observer])

# %%
# Treated and untreated tumours
# -----------------------------
# Untreated, the vessels the starved cells recruit should feed the periphery, the live cells should grow by orders of
# magnitude, and the radius should expand once the periphery is dense. Under the inhibitor the vessels and the live
# cells should stay orders of magnitude lower and the radius close to its initial 13 mm.

treated = trajectory
untreated = simulate(VascularCorePeripherySystem(dose=0.0), dt=0.5, solver=solver)
fig, (cells, vessels) = plt.subplots(1, 2, figsize=(10, 4))
for trajectory, name in ((untreated, 'untreated'), (treated, 'treated')):
    cells.semilogy(
        trajectory.times,
        trajectory.get_values('tumour.Ap') + trajectory.get_values('tumour.Ac'),
        label=f'live cells, {name}',
    )
    cells.semilogy(
        trajectory.times,
        trajectory.get_values('tumour.An').clip(1e-3),
        linestyle=':',
        label=f'necrotic, {name}',
    )
    vessels.semilogy(trajectory.times, trajectory.get_values('tumour.Vbp').clip(1e-6), label=f'periphery, {name}')
    vessels.semilogy(
        trajectory.times,
        trajectory.get_values('tumour.Vbc').clip(1e-6),
        linestyle=':',
        label=f'core, {name}',
    )
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
