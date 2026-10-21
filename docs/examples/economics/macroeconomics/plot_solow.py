"""
# Solow growth

Capital per worker grows when saving out of output exceeds depreciation, but output has diminishing returns to
capital, so growth slows and capital settles at a steady state where the two balance. Whatever the starting capital,
every economy with the same saving rate converges to the same steady state.
"""

# %%
# The production function is `y = k^alpha` with `alpha = 0.3`, the saving rate is `s = 0.2` and the depreciation rate
# `delta = 0.05`, so the steady state is `k* = (s / delta)^(1 / (1 - alpha))`, about 7.25. The economy starts with one
# unit of capital per worker, and the blueprint shows its approach to `k*`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'solow'
ALPHA, DELTA, UNTIL = 0.3, 0.05, 150

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, k=1.0, s=0.2), dt=1.0, observers=[observer]).run(until=UNTIL)

# %%
# Exact convergence and the golden rule
# -------------------------------------
# With `z = k^(1 - alpha)` the Solow equation becomes linear, `dz/dt = (1 - alpha) (s - delta z)`, so `z` relaxes to
# `s / delta` at the rate `(1 - alpha) delta`. Simulations from several starting capitals should follow this exact
# solution. Across saving rates, steady-state consumption `(1 - s) k*^alpha` should peak at the golden rule `s =
# alpha`.


def exact(k0, s, times):
    """Return the closed-form capital per worker of the Solow model."""
    z = s / DELTA + (k0 ** (1 - ALPHA) - s / DELTA) * np.exp(-(1 - ALPHA) * DELTA * times)
    return z ** (1 / (1 - ALPHA))


fig, (paths, golden) = plt.subplots(1, 2, figsize=(10, 4))
for index, k0 in enumerate((1.0, 4.0, 12.0)):
    trajectory = Simulation(create(MODEL, k=k0, s=0.2), dt=1.0).run(until=UNTIL).trajectory_
    paths.plot(trajectory.times[::5], trajectory['economy.k'][::5], 'o', ms=3, color=f'C{index}', label=f'k(0) = {k0}')
    paths.plot(trajectory.times, exact(k0, 0.2, trajectory.times), color=f'C{index}', lw=0.8)
paths.axhline((0.2 / DELTA) ** (1 / (1 - ALPHA)), color='grey', linestyle=':', label='k*')
paths.set_xlabel('time t')
paths.set_ylabel('capital per worker k')
paths.set_title('Simulated (dots) and exact (lines) paths')
paths.legend()
savings = np.linspace(0.05, 0.9, 18)
final = [Simulation(create(MODEL, k=1.0, s=s), dt=1.0).run(until=400).trajectory_['economy.k'][-1] for s in savings]
grid = np.linspace(0.02, 0.95, 200)
golden.plot(savings, (1 - savings) * np.array(final) ** ALPHA, 'o', label='simulation at t = 400')
golden.plot(grid, (1 - grid) * (grid / DELTA) ** (ALPHA / (1 - ALPHA)), 'k', lw=0.8, label='(1 - s) k*^alpha')
golden.axvline(ALPHA, color='C3', linestyle='--', label='s = alpha')
golden.set_xlabel('saving rate s')
golden.set_ylabel('steady-state consumption per worker')
golden.set_title('The golden rule')
golden.legend()
fig.tight_layout()
plt.show()
