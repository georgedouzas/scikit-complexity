"""
# Gauge arbitrage market

Exchange-rate potentials on the links of a lattice evolve by the discrete Maxwell equations, and traders move a good on
the sites down its covariant gradient. Their current is the source of the field, so the excess of the good stays equal
to the codifferential of the field at every time.
"""

# %%
# A run from a random density shows the good spreading and the field following it.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.finance import GaugeArbitrageMarketSystem

SYSTEM = 'gauge_arbitrage_market'

system = GaugeArbitrageMarketSystem()
observables = system.get_run_settings()['observables']
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, solver=RK4Solver(dt=0.02), observables=observables, observers=[observer])

# %%
# Conservation in the market
# --------------------------
# The Gauss residual stays at roundoff while the density relaxes.

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(trajectory.times, trajectory.values('gauss_residual') + 1e-16, label='Gauss residual')
ax.semilogy(trajectory.times, trajectory.values('sites.rho').std(axis=1), label='spread of the density')
ax.set_xlabel('time')
ax.set_title('The field tracks the good')
ax.legend()
plt.show()
