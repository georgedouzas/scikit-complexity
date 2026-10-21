"""
# Riccati prices

The price moves in proportion to the excess demand, with a linear demand and a supply that saturates quadratically.
The price equation is then a Riccati equation with two equilibrium prices: the lower one attracts every price below
the higher one, and prices above the higher one blow up in finite time.
"""

# %%
# The demand is `D = 10 - P` and the supply `S = 1 + 2 P - 0.1 P^2`, so `dP/dt = 9 - 3 P + 0.1 P^2`, with equilibria
# near 3.38 and 26.62. The price starts at 25, just below the unstable equilibrium, so it lingers there and then falls
# to the stable one, which the blueprint shows.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'riccati_prices'
GAMMA = -0.1
LOW, HIGH = np.sort(np.roots([-GAMMA, -3.0, 9.0]))

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, price=25.0), dt=0.02, observers=[observer]).run(until=4)

# %%
# Phase line and the exact solution
# ---------------------------------
# Writing `dP/dt = 0.1 (P - P1) (P - P2)` with the equilibria `P1 < P2`, the ratio `(P - P2) / (P - P1)` grows as
# `exp(0.1 (P2 - P1) t)`, which gives the price in closed form. Prices started anywhere below `P2` should follow it
# onto `P1`, and the phase line shows why prices above `P2` escape.


def exact(start, times):
    """Return the closed-form price of the Riccati equation."""
    ratio = (start - HIGH) / (start - LOW) * np.exp(-GAMMA * (HIGH - LOW) * times)
    return (HIGH - ratio * LOW) / (1 - ratio)


prices = np.linspace(-2.0, 30.0, 200)
fig, (phase, paths) = plt.subplots(1, 2, figsize=(10, 4))
phase.plot(prices, 9.0 - 3.0 * prices - GAMMA * prices**2, color='C0')
phase.axhline(0, color='grey', lw=0.8)
phase.plot([LOW], [0], 'o', color='black', label='stable P1')
phase.plot([HIGH], [0], 'o', mfc='white', color='black', label='unstable P2')
phase.set_xlabel('price P')
phase.set_ylabel('dP/dt')
phase.set_title('Phase line: P1 attracts, P2 repels')
phase.legend()
for index, start in enumerate((0.0, 10.0, 20.0, 25.0, 26.5)):
    run = Simulation(create(MODEL, price=start), dt=0.02).run(until=4).trajectory_
    paths.plot(run.times[::10], run['market.P'][::10], 'o', ms=3, color=f'C{index}', label=f'P(0) = {start}')
    paths.plot(run.times, exact(start, run.times), color=f'C{index}', lw=0.8)
paths.axhline(LOW, color='grey', linestyle=':')
paths.set_xlabel('time')
paths.set_ylabel('price P')
paths.set_title('Simulated (dots) and exact (lines) prices')
paths.legend(fontsize=8)
fig.tight_layout()
plt.show()
