"""
# Arbitrage traders

Schwichtenberg's gauge theory of arbitrage: traders carry a good on random walks along a line of markets, and the
exchange rates between neighbouring markets act as a connection that the density of the good couples to. Without an
arbitrage opportunity the good just diffuses, but a growing exchange-rate gradient, the field strength `E`, makes the
traders drift towards the profitable direction faster and faster.
"""

# %%
# The good starts as a Gaussian at the centre of a periodic line of 200 markets and length 20. The diffusion constant
# is `D = 0.5` and the arbitrage field strength `E = 0.5`. The blueprint shows the space-time history of the density,
# whose ridge bends into a parabola as the drift accelerates.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'arbitrage_traders'
N_SITES, LENGTH, D, E, UNTIL = 200, 20.0, 0.5, 0.5, 3.5

system = create(MODEL, n_sites=N_SITES, length=LENGTH, D=D, E=E)
blueprint = default_blueprint(system, primary='rho history')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=UNTIL)

# %%
# Drift, spreading, and growth
# ----------------------------
# Expanding `D (d_x - A_x)^2 rho` gives diffusion, a drift at the velocity `2 D A_x(t)` with `A_x(t) = E t`, and a
# growth at the rate `D A_x(t)^2`. The centre of the good should therefore move as `D E t^2`, its variance grow as
# `1 / 2 + 2 D t` as for free diffusion, and its total amount grow as `exp(D E^2 t^3 / 3)`. The drift is discretised
# with upwind differences, whose numerical diffusion `v dx / 2` adds `dx` times the distance travelled to the variance,
# the dashed correction.

trajectory = simulation.trajectory_
times = trajectory.times
spacing = LENGTH / N_SITES
positions = (np.arange(N_SITES) + 0.5) * spacing - LENGTH / 2
rho = trajectory['market.rho']
mass = rho.sum(axis=1) * spacing
centre = rho @ positions * spacing / mass
variance = rho @ positions**2 * spacing / mass - centre**2
fig, (moments, growth) = plt.subplots(1, 2, figsize=(10, 4))
moments.plot(times, centre, label='centre')
moments.plot(times, D * E * times**2, 'k--', lw=0.8)
moments.plot(times, variance, label='variance')
moments.plot(times, 0.5 + 2 * D * times, 'k:', lw=0.8, label='theory')
moments.plot(times, 0.5 + 2 * D * times + spacing * D * E * times**2, 'k--', lw=0.8, label='with upwind diffusion')
moments.set_xlabel('time t')
moments.set_ylabel('moment of the density')
moments.set_title('Accelerating drift, diffusive spreading')
moments.legend()
growth.semilogy(times, mass, label='total amount')
growth.semilogy(times, np.exp(D * E**2 * times**3 / 3), 'k--', lw=0.8, label='exp(D E^2 t^3 / 3)')
growth.set_xlabel('time t')
growth.set_ylabel('total amount of the good')
growth.set_title('Arbitrage creates value')
growth.legend()
fig.tight_layout()
plt.show()
