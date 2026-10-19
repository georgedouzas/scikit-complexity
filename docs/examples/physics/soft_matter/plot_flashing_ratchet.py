"""
# On-off flashing ratchet

Without any net force, the particles in the flashing sawtooth potential drift to the right: each time the potential
switches off they spread, and each time it switches back on more of them slide forward over the near maximum than
backward over the far one, so their positions climb from well to well.
"""

# %%
# The model is created by name from the catalog with 20 particles in a sawtooth of unit period and height that rises
# over a tenth of its period, at `kT = 0.05`, switched on and off at the rate 1, and simulated for 60 time units. The
# observer saves every frame with the default blueprint, whose primary view holds the positions of the particles.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'flashing_ratchet'

system = create(MODEL, n_particles=20, asymmetry=0.1, rate=1.0, duration=60.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.25, solver=EulerMaruyama(dt=0.004), observers=[observer], random_state=0)
simulation.run(until=60)

# %%
# Current against the asymmetry and the switching rate
# ----------------------------------------------------
# A symmetric sawtooth, `asymmetry = 0.5`, gives no current, and the current takes the sign of `0.5 - asymmetry`, so
# mirrored sawtooths drive opposite currents of the same size. The current vanishes for slow switching, where the
# particles equilibrate in both phases and every cycle is rare, and for fast switching, where they cannot relax into
# the wells, so it should peak at an intermediate rate. The error bars are three standard errors of the mean velocity
# of 200 particles.


def measure_current(asymmetry, rate, duration):
    run = create(MODEL, n_particles=200, asymmetry=asymmetry, rate=rate, duration=duration, random_state=1)
    positions = Simulation(run, dt=1.0, solver=EulerMaruyama(dt=0.004), random_state=2).run(until=duration)
    velocities = positions.trajectory_['particles.x'][-1, :, 0] / duration
    return velocities.mean(), 3 * velocities.std() / np.sqrt(velocities.size)


asymmetries = np.array([0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
by_asymmetry = np.array([measure_current(asymmetry, 1.0, 40.0) for asymmetry in asymmetries])
rates = np.array([0.25, 1.0, 4.0, 16.0, 64.0])
by_rate = np.array([measure_current(0.1, rate, min(40.0, 120.0 / rate)) for rate in rates])

fig, (sign, optimum) = plt.subplots(1, 2, figsize=(10, 4))
sign.errorbar(asymmetries, by_asymmetry[:, 0], yerr=by_asymmetry[:, 1], fmt='o')
sign.axhline(0.0, color='grey', linestyle='--')
sign.axvline(0.5, color='grey', linestyle=':', label='symmetric sawtooth')
sign.set_xlabel('asymmetry of the sawtooth')
sign.set_ylabel('mean velocity')
sign.set_title('The current follows the asymmetry')
sign.legend()
optimum.errorbar(rates, by_rate[:, 0], yerr=by_rate[:, 1], fmt='o-')
optimum.set_xscale('log')
optimum.set_xticks(rates, [f'{rate:g}' for rate in rates])
optimum.minorticks_off()
optimum.set_xlabel('switching rate')
optimum.set_ylabel('mean velocity')
optimum.set_title('An optimal switching rate')
fig.tight_layout()
plt.show()
