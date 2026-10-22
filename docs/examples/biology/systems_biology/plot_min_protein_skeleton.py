"""
# Min-protein skeleton model

In the bacterium E. coli, the proteins MinD and MinE cycle between the cell membrane and the cytosol and form patterns
that mark the middle of the cell. MinD recruits more MinD and MinE to the membrane, and MinE detaches MinD again, while
the totals of both proteins stay fixed. The space-time image shows membrane MinD on a periodic line, one row per saved
step with the earliest at the top: after a uniform start, narrow bands of MinD travel along the line, meet head-on in
bright spots, and new bands leave every meeting point in the opposite directions.
"""

# %%
# The model is created by name from the catalog on a periodic line of length 100 with 128 sites, with the mean total
# densities 200 of MinD and 120 of MinE and 1% noise on membrane MinD. The catalog registers the implicit `BDF` method
# with the exact Jacobian, since the fast cytosolic diffusion makes the system stiff. The default blueprint shows every
# state as a space-time image, with membrane MinD first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import dispersion_relation
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'min_protein_skeleton'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=4.0,
    solver=ScipyODE(method='BDF', jacobian='exact'),
    observers=[observer],
    random_state=0,
).run(until=400)

# %%
# Oscillatory instability and conservation
# ----------------------------------------
# The reactions only move proteins between states, so the totals `md + mde + cdt + cdd` of MinD and `mde + ce` of
# MinE should stay at their initial values. The uniform steady state is unstable in a band of long waves whose modes
# also turn, which is the oscillatory instability of the Min system (Brauns and Marchetti, App. F): the dispersion
# relation of the linearised laws should show positive growth rates with nonzero frequencies.

trajectory = simulation.trajectory_
mind_total = sum(trajectory[f'cell.{name}'] for name in ('md', 'mde', 'cdt', 'cdd')).sum(axis=1)
mine_total = sum(trajectory[f'cell.{name}'] for name in ('mde', 'ce')).sum(axis=1)
relation = dispersion_relation(create(MODEL, n_sites=512, length=400.0))
order = np.argsort(relation.magnitudes)

fig, (rates, totals) = plt.subplots(1, 2, figsize=(10, 4))
rates.plot(relation.magnitudes[order], relation.growth_rates[order], label='growth rate')
rates.plot(relation.magnitudes[order], np.abs(relation.frequencies[order]), '--', label='|frequency|')
rates.axhline(0.0, color='gray', linewidth=0.8)
rates.set_xlim(0, 1.0)
rates.set_ylim(-0.2, 0.2)
rates.set_xlabel('wavenumber q')
rates.set_ylabel('rate (1/s)')
rates.set_title('An oscillatory band of long waves is unstable')
rates.legend()
totals.plot(trajectory.times, mind_total / mind_total[0] - 1, label='MinD')
totals.plot(trajectory.times, mine_total / mine_total[0] - 1, label='MinE')
totals.set_xlabel('time (s)')
totals.set_ylabel('relative change of the total')
totals.set_title('Both proteins are conserved')
totals.legend()
fig.tight_layout()
plt.show()
