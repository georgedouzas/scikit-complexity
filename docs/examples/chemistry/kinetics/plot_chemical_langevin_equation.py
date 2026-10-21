"""
# Chemical Langevin equation

Schlögl's reaction has a low stable state near 85 molecules of `X` and a high one near 570. Started at the unstable
state between them, the intrinsic noise of the reactions decides which one a run reaches, and the run then
fluctuates around it.
"""

# %%
# The model is created by name from the catalog with Gillespie's rate constants and starts at `x = 250`, near the
# unstable state. It is integrated with the Euler-Maruyama scheme for ten time units. The
# default blueprint shows `x` in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'chemical_langevin_equation'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
solver = EulerMaruyama(dt=0.001)
simulation = Simulation(system, dt=0.05, solver=solver, observers=[observer], random_state=0).run(until=10)

# %%
# Splitting between the two states
# --------------------------------
# The deterministic drift `c1 A x (x - 1) / 2 - c2 x (x - 1) (x - 2) / 6 + c3 B - c4 x` has two stable roots and an
# unstable one between them. Runs started at the same `x = 250` with different seeds should split between the two
# stable roots, so the histogram of their final states should have two peaks at them and nothing near the unstable
# root.

c1, c2, c3, c4, A, B = 3e-7, 1e-4, 1e-3, 3.5, 1e5, 2e5
drift = np.polynomial.Polynomial([c3 * B, -c1 * A / 2 - c4 - c2 / 3, c1 * A / 2 + c2 / 2, -c2 / 6])
roots = np.sort(drift.roots().real)
fig, (runs, finals) = plt.subplots(1, 2, figsize=(10, 4))
ends = []
for seed in range(30):
    run = Simulation(create(MODEL), dt=0.05, solver=solver, random_state=seed + 1).run(until=10).trajectory_
    ends.append(run['reactor.x'][-1])
    runs.plot(run.times, run['reactor.x'], color='C1' if ends[-1] > roots[1] else 'C0', linewidth=0.6, alpha=0.6)
for root in roots:
    runs.axhline(root, color='black', linestyle='--', linewidth=0.8)
runs.set_xlabel('time')
runs.set_ylabel('molecules x')
runs.set_title('30 runs from the unstable state')
edges, density = histogram(ends, bins=20)
finals.stairs(density, edges, fill=True, alpha=0.5, label='final x')
for root, label in zip(roots, ['stable', 'unstable', 'stable'], strict=True):
    finals.axvline(root, color='black', linestyle='--' if label == 'stable' else ':', linewidth=1)
finals.set_xlabel('molecules x at t = 10')
finals.set_ylabel('probability density')
finals.set_title('Final states cluster at the two stable roots')
fig.tight_layout()
plt.show()
