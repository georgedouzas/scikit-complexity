"""
# Optimal harvesting

The owner of a logistic fishery chooses the harvest effort that maximises the discounted profit. The optimal effort is
bang-bang with a singular arc: it fishes at full effort until the stock falls to the level where the yield net of costs
earns the discount rate, holds the stock there with a steady effort, and fishes at full effort again at the end, when
the stock left over has no value.
"""

# %%
# The model is created by name from the catalog with the growth rate and the capacity 1, the cost 0.2 per unit of
# effort, and the discount rate 0.1, from a full stock. The collocation solver of its catalog entry transcribes the
# problem on 101 nodes over 20 years. The default blueprint shows the stock and the effort in one time series view.

import matplotlib.pyplot as plt
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'optimal_harvesting'

system = create(MODEL)
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=0.2,
    solver=entry.defaults['solver'],
    observers=[observer],
    observables=entry.observables_of(system),
).run(until=20)

# %%
# The singular level
# ------------------
# Between the two arcs at full effort, the stock should sit at the level `x*` of Clark's modified golden rule,
# `r (1 - 2 x*) + c r (1 - x*) / (x* - c) = delta`, and the effort should be the one that holds it there,
# `r (1 - x*)`.

level = brentq(lambda x: 1 - 2 * x + 0.2 * (1 - x) / (x - 0.2) - 0.1, 0.21, 1.0)
trajectory = simulation.trajectory_
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(trajectory.times, trajectory['resource.x'], label='stock x')
ax.plot(trajectory.times, trajectory['resource.effort'], label='effort E')
ax.axhline(level, color='black', linestyle='--', linewidth=0.8, label=f'singular level x* = {level:.3f}')
ax.axhline(1 - level, color='gray', linestyle=':', linewidth=0.8, label=f'singular effort 1 - x* = {1 - level:.3f}')
ax.set_xlabel('time (years)')
ax.set_ylabel('stock and effort')
ax.set_title('Bang-bang effort around a singular arc')
ax.legend()
plt.show()
