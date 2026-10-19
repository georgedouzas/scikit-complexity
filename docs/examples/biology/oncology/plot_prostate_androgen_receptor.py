"""
# Prostate cancer driven by activated androgen receptors

With high receptor activation the androgen-dependent cancer cells grow towards their capacity and the PSA in the
tissue and the serum rises with them.
"""

# %%
# The model is created by name from the catalog: a 60 year old man with 1.5 billion normal epithelial cells and one
# billion cancer cells, at receptor activations of 40 nM, simulated for three years in hours. The observer saves every
# frame with the default blueprint, whose primary view is the time series of the cells and the PSA.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'prostate_androgen_receptor'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=168.0, solver=ScipyODE(method='LSODA'), observers=[observer]).run(until=26280)

# %%
# Androgen ablation
# -----------------
# Lowering both activations to 5 nM, as an androgen ablation does, makes the death rate of the dependent cells exceed
# their growth rate, so the cancer should shrink and the serum PSA fall instead of rising.

trajectory = simulation.trajectory_
ablated = Simulation(create(MODEL, UT=5.0, UD=5.0), dt=168.0, solver=ScipyODE(method='LSODA')).run(until=26280)
ablation = ablated.trajectory_
years = trajectory.times / 8760
fig, (cells, psa) = plt.subplots(1, 2, figsize=(10, 4))
cells.plot(years, trajectory['prostate.AD'], label='dependent, normal activation')
cells.plot(years, ablation['prostate.AD'], label='dependent, ablation')
cells.plot(years, trajectory['prostate.N'], label='normal epithelium')
cells.set_xlabel('time (years)')
cells.set_ylabel('cells (billions)')
cells.set_title('Ablation shrinks the cancer')
cells.legend()
psa.plot(years, trajectory['prostate.UP'], label='normal activation')
psa.plot(years, ablation['prostate.UP'], label='ablation')
psa.set_xlabel('time (years)')
psa.set_ylabel('serum PSA (ng/mL)')
psa.set_title('Serum PSA follows the cancer')
psa.legend()
fig.tight_layout()
plt.show()
