"""
# TCR gene therapy

The effector cells and the tumour cells swing in damped predator-prey cycles and settle on a dormant tumour of about
twelve thousand cells, far below the carrying capacity of a billion.
"""

# %%
# The model is created by name from the catalog with the book's parameters and a thousand effector and tumour cells,
# and runs for 1500 days. The observer saves every second frame with the default blueprint, whose primary view is the
# time series of the two populations.

import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tcr_gene_therapy'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=5.0, observers=[observer]).run(until=1500)

# %%
# Spiral into the dormant state
# -----------------------------
# The equilibrium solves `s + alpha A - d1 E + r1 E / (E + g1) = 0` and `r2 (1 - b A) = a E / (A + g2)`. In the plane of
# the tumour and the effector cells the trajectory should spiral into that point.


def rates(y):
    E, A = y
    return [1 + 0.05 * A - 0.03 * E + 0.1245 * E / (E + 1e-3), 0.18 * A * (1 - 1e-9 * A) - E * A / (A + 1e5)]


E, A = simulation.trajectory_['site.E'], simulation.trajectory_['site.A']
E_star, A_star = fsolve(rates, [E[-1], A[-1]])
fig, (series, plane) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, A, label='tumour cells')
series.plot(simulation.trajectory_.times, E, label='effector cells')
series.axhline(A_star, color='grey', linestyle=':', label='dormant tumour')
series.set_xlabel('time (days)')
series.set_ylabel('cells')
series.set_title('Damped tumour-immune cycles')
series.legend()
plane.plot(A, E, label='simulated')
plane.plot(A_star, E_star, 'o', color='black', label='equilibrium')
plane.set_xlabel('tumour cells')
plane.set_ylabel('effector cells')
plane.set_title('The trajectory spirals into the equilibrium')
plane.legend()
fig.tight_layout()
plt.show()
