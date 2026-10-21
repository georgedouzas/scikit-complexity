"""
# Capital stock

Meadows' simplest stock-and-flow economy: capital produces output, a fraction of the output is reinvested as new
capital, and capital wears out over its lifetime. The reinforcing loop of investment and the balancing loop of
depreciation compete, and whichever dominates makes the stock grow or decay exponentially.
"""

# %%
# Capital produces a third of its value as output each year, lasts fifteen years, and a quarter of the output is
# reinvested. Investment then outweighs depreciation, and the blueprint shows the capital stock growing from 100 over
# fifty years.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'capital_stock'
OUTPUT, LIFETIME, UNTIL = 1 / 3, 15.0, 50.0

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
system = create(MODEL, capital=100.0, investment_fraction=0.25, output_per_capital=OUTPUT, lifetime=LIFETIME)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=UNTIL)

# %%
# Loop dominance
# --------------
# The stock follows `dK/dt = (f o - 1 / L) K`, so it changes exponentially at the net rate `f o - 1 / L`. The stock is
# in equilibrium at the investment fraction `f = 1 / (o L) = 0.2`, grows above it, and decays below it, and every
# simulated stock should follow its exponential.

fig, (paths, rates) = plt.subplots(1, 2, figsize=(10, 4))
fractions = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
measured = []
for index, fraction in enumerate(fractions):
    system = create(MODEL, capital=100.0, investment_fraction=fraction, output_per_capital=OUTPUT, lifetime=LIFETIME)
    trajectory = Simulation(system, dt=1.0).run(until=UNTIL).trajectory_
    rate = fraction * OUTPUT - 1 / LIFETIME
    paths.semilogy(
        trajectory.times[::3],
        trajectory['economy.K'][::3],
        'o',
        ms=3,
        color=f'C{index}',
        label=f'f = {fraction}',
    )
    paths.semilogy(trajectory.times, 100 * np.exp(rate * trajectory.times), color=f'C{index}', lw=0.8)
    measured.append(np.log(trajectory['economy.K'][-1] / 100) / UNTIL)
paths.set_xlabel('year')
paths.set_ylabel('capital stock K')
paths.set_title('Exponential growth or decay')
paths.legend(fontsize=8)
grid = np.linspace(0.05, 0.35, 2)
rates.plot(fractions, measured, 'o', label='simulation')
rates.plot(grid, grid * OUTPUT - 1 / LIFETIME, 'k', lw=0.8, label='f o - 1 / L')
rates.axhline(0, color='grey', lw=0.8)
rates.axvline(1 / (OUTPUT * LIFETIME), color='C3', linestyle='--', label='equilibrium f = 0.2')
rates.set_xlabel('investment fraction f')
rates.set_ylabel('net growth rate')
rates.set_title('Investment against depreciation')
rates.legend()
fig.tight_layout()
plt.show()
