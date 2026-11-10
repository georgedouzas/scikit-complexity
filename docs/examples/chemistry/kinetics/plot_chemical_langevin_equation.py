"""
# Chemical Langevin equation

Schlögl's reaction has a low stable state near 17 molecules of `X` and a high one near 113. In a small volume the
intrinsic noise of the reactions is strong enough to carry the system across the barrier between them, so it
fluctuates around one state for a while and then switches to the other, again and again.
"""

# %%
# The example builds `ChemicalLangevinEquationSystem` with its default rate constants, Gillespie's scaled to a volume
# five times smaller, and starts at `x = 50`. It is integrated with the Euler-Maruyama scheme for 200 time units. The
# default blueprint shows `x` in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.kinetics import ChemicalLangevinEquationSystem

SYSTEM = 'chemical_langevin_equation'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=20)
trajectory = simulate(ChemicalLangevinEquationSystem(), observers=[observer], random_state=0)

# %%
# Two states and the switches between them
# ----------------------------------------
# The deterministic drift `c1 A x (x - 1) / 2 - c2 x (x - 1) (x - 2) / 6 + c3 B - c4 x` has two stable roots and an
# unstable one between them. The run should spend its time near the two stable roots, so the histogram of `x` should
# have two peaks at them and a dip at the unstable root, which the switches cross.

c1, c2, c3, c4, A, B = 1.5e-6, 2.5e-3, 2e-4, 3.5, 1e5, 2e5
drift = np.polynomial.Polynomial([c3 * B, -c1 * A / 2 - c4 - c2 / 3, c1 * A / 2 + c2 / 2, -c2 / 6])
roots = np.sort(drift.roots().real)
x = trajectory.get_values('reactor.x')
histogram = distribution(x[trajectory.times > 5], kind='histogram', bins=40)
edges, density = histogram.edges, histogram.density
fig, (series, distribution) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times, x, linewidth=0.5)
for root in roots:
    series.axhline(root, color='black', linestyle='--', linewidth=0.8)
series.set_xlabel('time')
series.set_ylabel('molecules x')
series.set_title('Noise-driven switches between two states')
distribution.stairs(density, edges, fill=True, alpha=0.5)
for root, label in zip(roots, ['stable', 'unstable', 'stable'], strict=True):
    distribution.axvline(root, color='black', linestyle='--' if label == 'stable' else ':', linewidth=1)
distribution.set_xlabel('molecules x')
distribution.set_ylabel('probability density')
distribution.set_title('Two peaks at the stable states')
fig.tight_layout()
plt.show()
