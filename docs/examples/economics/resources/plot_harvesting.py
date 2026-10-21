"""
# Harvesting

A socio-ecological co-simulation: a village of farmer agents harvests a resource that regrows logistically, coupled
to it through the size of the village. Each farmer takes a fixed share of the stock, so the stock settles where
regrowth balances the harvest, lower the more farmers there are, and collapses once the total effort exceeds the
growth rate. The sustained harvest peaks at an intermediate village size, the maximum sustainable yield.
"""

# %%
# Ten farmers each harvest 5 percent of the stock a year from a resource with growth rate `r = 1` and carrying
# capacity `K = 1000`. The total effort `0.5 = r / 2` is the effort of the maximum sustainable yield, and the
# blueprint shows the stock falling from its carrying capacity to the equilibrium `K (1 - E / r) = 500`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'harvesting'
EFFORT, R, K, UNTIL = 0.05, 1.0, 1000.0, 20

system = create(MODEL, farmers=10, effort=EFFORT, r=R, K=K)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=default_blueprint(system), interval=1)
simulation = Simulation(system, observers=[observer]).run(until=UNTIL)

# %%
# Equilibrium stock and sustainable yield
# ---------------------------------------
# With the total effort `E = e F` of `F` farmers, the logistic stock settles at `K (1 - E / r)` while `E < r` and dies
# out otherwise, and the sustained harvest `E K (1 - E / r)` is largest at `E = r / 2`, ten farmers here. The final
# stock and harvest of villages of every size should follow these curves.

villages = np.arange(1, 26, 2)
stocks = np.array(
    [
        Simulation(create(MODEL, farmers=int(farmers), effort=EFFORT, r=R, K=K))
        .run(until=200)
        .trajectory_['land/stock.n'][-1]
        for farmers in villages
    ],
)
grid = np.linspace(0, 25, 200)
equilibrium = K * np.maximum(1 - EFFORT * grid / R, 0)
fig, (stock, harvest) = plt.subplots(1, 2, figsize=(10, 4))
stock.plot(villages, stocks, 'o', label='simulation after 200 years')
stock.plot(grid, equilibrium, 'k', lw=0.8, label='K (1 - e F / r)')
stock.set_xlabel('number of farmers F')
stock.set_ylabel('final stock')
stock.set_title('More farmers, a smaller stock')
stock.legend()
harvest.plot(villages, EFFORT * villages * stocks, 'o', label='simulation')
harvest.plot(grid, EFFORT * grid * equilibrium, 'k', lw=0.8, label='e F K (1 - e F / r)')
harvest.axvline(R / (2 * EFFORT), color='C3', linestyle='--', label='maximum sustainable yield')
harvest.set_xlabel('number of farmers F')
harvest.set_ylabel('sustained harvest per year')
harvest.set_title('The yield peaks at E = r / 2')
harvest.legend()
fig.tight_layout()
plt.show()
