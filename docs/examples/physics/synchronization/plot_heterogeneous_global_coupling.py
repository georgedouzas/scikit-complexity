"""
# Heterogeneous global coupling

Fifty oscillators with time scales spread between 0.2 and 1.8 turn on their limit cycles until the global coupling
`k = 2` is switched on, and then all of them come to rest at the fixed point that each one alone leaves. Without the
spread of time scales the coupling could not stabilize it.
"""

# %%
# Each system is the Hopf normal form with the unstable focus `0.1 ± 2i`, sped up or slowed down by its time scale. The
# schedule keeps the systems uncoupled for 15 time units. The default blueprint shows the variables `x_j` of all
# systems as time series.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'heterogeneous_global_coupling'

system = create(MODEL, n_systems=50, k=2.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)


def coupling_at(time):
    """Return no coupling for the first 15 time units and the coupling 2 afterwards."""
    return 0.0 if time < 15.0 else 2.0


simulation = Simulation(system, dt=0.2, observers=[observer], schedule={'coupling.K': coupling_at}).run(until=40)

# %%
# The window of stability
# -----------------------
# The order parameters `X`, `Y`, `V`, and `W` close on four linear equations near the fixed point, whose eigenvalues
# `nu - k / 2 ± √(k² / 4 + sigma² nu²)` for `nu = 0.1 ± 2i` decide its stability. Their largest real part should be
# negative only in a window of coupling, roughly `2 beta (1 + sigma) < k < (gamma² - beta²) sigma² / beta`. Inside the
# window, the decay rate of the mean amplitude of runs at each coupling should follow that eigenvalue, closely towards
# the upper end and roughly near its minimum, where the neglected terms of order `δτ² δx` matter. Outside the window,
# the oscillations saturate and the measured rate is 0. At the weakest coupling the closure, which assumes that every
# system stays close to the mean, does not hold, and the fixed point is stable once `k` exceeds the largest growth rate
# `beta max(tau)` of a single system.

beta, gamma = 0.1, 2.0
sigma = float(np.std(system.elements[0].values['tau']))
nu = complex(beta, gamma)
grid = np.linspace(0.01, 16.0, 400)
leading = [max((nu - k / 2 + sign * np.sqrt(k**2 / 4 + sigma**2 * nu**2)).real for sign in (1, -1)) for k in grid]
couplings = [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
rates = []
for k in couplings:
    run = Simulation(create(MODEL, n_systems=50, k=k), dt=0.5).run(until=100).trajectory_
    times, amplitude = np.asarray(run.times), np.asarray(run['amplitude'])
    late = (times >= 10.0) & (amplitude > 1e-8)
    rates.append(np.polyfit(times[late], np.log(amplitude[late]), 1)[0])
fig, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(2 * beta * (1 + sigma), (gamma**2 - beta**2) * sigma**2 / beta, alpha=0.15, label='estimated window')
ax.plot(grid, leading, color='black', label='order-parameter equations')
ax.plot(couplings, rates, 'o', label='simulated decay rate')
ax.axhline(0.0, color='grey', linewidth=0.8)
ax.set_xlabel('coupling k')
ax.set_ylabel(r'largest Re $\lambda$')
ax.set_title(f'Restabilization of the fixed point, sigma = {sigma:.2f}')
ax.legend()
plt.show()
