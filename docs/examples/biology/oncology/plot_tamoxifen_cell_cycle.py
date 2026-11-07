"""
# Gompertzian cell-cycle phases under tamoxifen

A continuous infusion of tamoxifen and its metabolite kills cells in the growth phase, and the tumour shrinks from
1000 mm³ while the drug levels settle.
"""

# %%
# The system is built from its class: a breast tumour of 900 mm³ in the growth phase and 50 mm³ each in synthesis and
# mitosis receives 300 μg of tamoxifen a day. The run spans 100 days in hours. The observer saves every second frame
# with the default blueprint, whose primary view is the time series of the phases and the drug masses.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import TamoxifenCellCycleSystem

SYSTEM = 'tamoxifen_cell_cycle'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = simulate(TamoxifenCellCycleSystem(), solver=ScipyODESolver(method='LSODA'), observers=[observer])

# %%
# Treated and untreated
# ---------------------
# Without the drug the tumour grows towards `Asat = 10000` mm³ and the lymphocytes stay at `s / d1`. Under tamoxifen
# the tumour should shrink and the lymphocytes settle lower, at `s / (d1 + a2 (UX + delta2 UH))`.

trajectory = simulation
untreated = simulate(TamoxifenCellCycleSystem(u=0.0), solver=ScipyODESolver(method='LSODA'))
days = trajectory.times / 24


def volume(run):
    """Return the total tumour volume of a run."""
    return run.values('tumour.Ag') + run.values('tumour.As') + run.values('tumour.Am')


exposure = trajectory.values('tumour.X2')[-1] / 8.592 + 25 * trajectory.values('tumour.X3')[-1] / 8.592
fig, (tumour, lymphocytes) = plt.subplots(1, 2, figsize=(10, 4))
tumour.plot(days, volume(trajectory), label='tamoxifen')
tumour.plot(days, volume(untreated), label='untreated')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour volume (mm³)')
tumour.set_title('Tamoxifen shrinks the tumour')
tumour.legend()
lymphocytes.plot(days, trajectory.values('tumour.E'), label='tamoxifen')
lymphocytes.plot(days, untreated.values('tumour.E'), label='untreated')
lymphocytes.axhline(0.121 / (0.012 + 0.010 * exposure), color='black', linestyle='--', label='treated balance')
lymphocytes.set_xlabel('time (days)')
lymphocytes.set_ylabel('circulating lymphocytes (millions)')
lymphocytes.set_title('The drug also lowers the lymphocytes')
lymphocytes.legend()
fig.tight_layout()
plt.show()
