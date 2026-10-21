"""
# Ott-Antonsen

The order parameter of infinitely many Kuramoto oscillators obeys one exact equation: above the critical coupling
`K_c = 2 Δ` a small coherence grows along a sigmoid and settles at `r = sqrt(1 - 2 Δ / K)`.
"""

# %%
# The order parameter starts at `r = 0.1` with the coupling `K = 2` and the half width `Δ = 0.5` of the Lorentzian
# frequencies, so it settles at `sqrt(1/2)`. The default blueprint shows the real and imaginary parts of the mean field.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'ott_antonsen'

system = create(MODEL, r=0.1, K=2.0, Delta=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# Exact growth and the bifurcation diagram
# ----------------------------------------
# With `ω = 0` the modulus obeys `r' = (K / 2 - Δ) r - K r^3 / 2`, a logistic equation for `r^2` with the solution
# `r(t)^2 = r∞^2 / (1 + (r∞^2 / r0^2 - 1) exp(-(K - 2 Δ) t))`. The simulated growth should lie on it, and the final
# order parameters of a sweep of `K` should follow `sqrt(1 - 2 Δ / K)` above `K_c = 1` and vanish below.

times = np.asarray(simulation.trajectory_.times)
r = np.hypot(np.ravel(simulation.trajectory_['field.x']), np.ravel(simulation.trajectory_['field.y']))
final = 1 - 2 * 0.5 / 2.0
exact = np.sqrt(final / (1 + (final / 0.1**2 - 1) * np.exp(-(2.0 - 1.0) * times)))
couplings = np.linspace(0.2, 5.0, 25)
settled = []
for coupling in couplings:
    run = Simulation(create(MODEL, r=0.1, K=coupling, Delta=0.5), dt=0.5).run(until=200)
    settled.append(np.hypot(np.ravel(run.trajectory_['field.x'])[-1], np.ravel(run.trajectory_['field.y'])[-1]))
grid = np.linspace(1.0, 5.0, 200)
fig, (growth, diagram) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(times, r, label='simulated')
growth.plot(times, exact, '--', color='black', label='exact')
growth.set_xlabel('time')
growth.set_ylabel('order parameter r')
growth.set_title('Growth of coherence at K = 2')
growth.legend()
diagram.plot(couplings, settled, 'o', label='simulated')
diagram.plot(grid, np.sqrt(1 - 1.0 / grid), color='black', label='sqrt(1 - 2Δ / K)')
diagram.set_xlabel('coupling K')
diagram.set_ylabel('final order parameter')
diagram.set_title('Supercritical onset at K_c = 2Δ = 1')
diagram.legend()
fig.tight_layout()
plt.show()
