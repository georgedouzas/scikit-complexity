"""
# Trastuzumab and paclitaxel against HER2+ breast cancer

The antibody binds the receptors at once, and from day 2 the tumour cells grow toward their carrying capacity at half
their rate, while the paclitaxel accumulates in the cells.
"""

# %%
# The model is created by name from the catalog with the culture of Jarrett et al.: 35,000 cells, counted in millions,
# under a free antibody and a free paclitaxel from day 1. The observer saves every frame with the default blueprint,
# whose primary view is the time series of the cells, the bound antibody, and the internalized paclitaxel.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'her2_targeted_therapy'
SOLVER = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.05, solver=SOLVER, observers=[observer]).run(until=10)

# %%
# Logistic growth at a lowered rate
# ---------------------------------
# The bound antibody equals the free antibody of 0.2, so from day 2 the growth rate is `r (1 - a1 Ab) = 0.41` a day
# instead of 0.83. The treated tumour should follow the logistic curve of that rate, and lag behind the untreated one.

untreated = Simulation(create(MODEL, antibody=0.0, paclitaxel=0.0), dt=0.05, solver=SOLVER).run(until=10).trajectory_
treated = simulation.trajectory_
times, cells = treated.times, treated['tumour.A']
after = times >= 2.0
rate = 0.83 * (1 - 2.51 * 0.2)
logistic = 0.67 / (1 + (0.67 / cells[after][0] - 1) * np.exp(-rate * (times[after] - 2.0)))
fig, (growth, drugs) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(untreated.times, untreated['tumour.A'], label='untreated')
growth.plot(times, cells, label='trastuzumab and paclitaxel')
growth.plot(times[after], logistic, '--', color='black', label='logistic at rate 0.41')
growth.set_xlabel('time (days)')
growth.set_ylabel('cells (millions)')
growth.set_title('The antibody halves the growth rate')
growth.legend()
drugs.plot(times, treated['tumour.Ab'], label='bound antibody')
drugs.plot(times, treated['tumour.Pin'], label='internalized paclitaxel (pmol)')
drugs.set_xlabel('time (days)')
drugs.set_ylabel('amount')
drugs.set_title('Binding is fast and uptake is slow')
drugs.legend()
fig.tight_layout()
plt.show()
