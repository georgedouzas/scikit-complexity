"""
# Cobweb

Producers decide their supply on the previous price, and the price then clears the market against demand. The price
overshoots the equilibrium in alternate directions, and the oscillation dies out when supply is less sensitive to the
price than demand, the cobweb theorem of Ezekiel.
"""

# %%
# Demand is `a - b p` and supply is `c + d p_prev`, so the price follows the map `p' = (a - c - d p) / b` with the
# equilibrium `p* = (a - c) / (b + d) = 2.5`. With `b = 2` and `d = 1.6` the oscillation is slowly damped. The blueprint
# shows the price over thirty periods.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cobweb'
A, B, C = 10.0, 2.0, 1.0

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, price=1.0, a=A, b=B, c=C, d=1.6), dt=1, observers=[observer]).run(until=30)

# %%
# Cobweb diagram and the cobweb theorem
# -------------------------------------
# The price alternates around `p*`, and its distance from `p*` is multiplied by `-d / b` every period. The deviation
# should therefore shrink geometrically for `d < b`, stay constant for `d = b`, and grow for `d > b`, along the lines
# `(d / b)^t`.


def prices(d, until=30):
    """Return the prices of a cobweb market with supply slope `d`."""
    return Simulation(create(MODEL, price=1.0, a=A, b=B, c=C, d=d), dt=1).run(until=until).trajectory_['market.price']


price = simulation.trajectory_['market.price']
grid = np.linspace(0.5, 5.0, 2)
fig, (cobweb, theorem) = plt.subplots(1, 2, figsize=(10, 4))
cobweb.plot(grid, (A - C - 1.6 * grid) / B, color='C0', label='price map')
cobweb.plot(grid, grid, color='grey', linestyle=':', label='p\' = p')
corners = np.repeat(price, 2)
cobweb.plot(corners[:-1], corners[1:], color='C1', lw=0.8, label='trajectory')
cobweb.set_xlabel('price p')
cobweb.set_ylabel('next price p\'')
cobweb.set_title('Cobweb diagram for d = 1.6')
cobweb.legend()
times = np.arange(31)
for d, color in ((1.0, 'C0'), (2.0, 'C2'), (2.2, 'C3')):
    equilibrium = (A - C) / (B + d)
    theorem.semilogy(times, np.abs(prices(d) - equilibrium), 'o', ms=3, color=color, label=f'd / b = {d / B:.1f}')
    theorem.semilogy(times, abs(1.0 - equilibrium) * (d / B) ** times, color=color, lw=0.8)
theorem.set_xlabel('period t')
theorem.set_ylabel('|p - p*|')
theorem.set_title('The deviation scales as (d / b)^t')
theorem.legend()
fig.tight_layout()
plt.show()
