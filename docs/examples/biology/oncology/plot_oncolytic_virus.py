"""
# Oncolytic virus

The infected cells spread through the tumour, and the uninfected and infected cells settle on a coexistence point at
two thirds of the carrying capacity: the virus reduces the tumour but does not eradicate it.
"""

# %%
# The model is created by name from the catalog, from a tenth of the capacity of uninfected cells and a trace of
# infected cells, and runs for 60 growth times. The observer saves every second frame with the default blueprint,
# whose primary view is the time series of the two populations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'oncolytic_virus'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=60)

# %%
# Coexistence point
# -----------------
# For `alpha = 1`, `r = 0.5`, and `d = 0.5` the nullclines `1 - AU - AV = alpha AV` and
# `r (1 - AU - AV) + alpha AU = d` cross at `AV = (1 - d / alpha) / (1 + alpha - r) = 1/3` and
# `AU = d / alpha - r AV = 1/3`. The trajectory should end on that point.

alpha, r, d = 1.0, 0.5, 0.5
AU, AV = simulation.trajectory_['tumour.AU'], simulation.trajectory_['tumour.AV']
grid = np.linspace(0, 1, 50)
fig, (series, plane) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, AU, label='uninfected')
series.plot(simulation.trajectory_.times, AV, label='infected')
series.plot(simulation.trajectory_.times, AU + AV, '--', label='whole tumour')
series.set_xlabel('time')
series.set_ylabel('cells (fraction of capacity)')
series.set_title('The virus spreads and the tumour shrinks')
series.legend()
plane.plot(AU, AV, label='simulated')
plane.plot(grid, (1 - grid) / (1 + alpha), ':', label='uninfected nullcline')
plane.plot(grid, 1 - grid + (alpha * grid - d) / r, ':', label='infected nullcline')
plane.plot(1 / 3, 1 / 3, 'o', color='black', label='coexistence point')
plane.set_xlim(0, 1)
plane.set_ylim(0, 1)
plane.set_xlabel('uninfected cells')
plane.set_ylabel('infected cells')
plane.set_title('The trajectory ends on the coexistence point')
plane.legend()
fig.tight_layout()
plt.show()
