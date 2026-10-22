"""
# Curie-Weiss model

Below the critical temperature `T_c = J`, 200 fully connected spins that start at random order within a few units of
time. The space-time image shows the spins drawn in a row, one row per frame with the earliest at the top, turning
to one color with a few short-lived flips of single spins.
"""

# %%
# The model is created by name from the catalog with 200 spins at `T = 0.5 J`, starting from random spins. One unit
# of time is one attempted flip per spin. The observer saves every frame with the default blueprint, whose primary
# view is the space-time image of the spins.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'curie_weiss'

system = create(MODEL, n_spins=200, temperature=0.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=0.2,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=30)

# %%
# Magnetization against temperature
# ---------------------------------
# The mean-field theory of Bertin's section 1.4.1 gives the magnetization as the positive root of
# `m = tanh(m J / T)` below `T_c = J` and zero above it. The simulated mean absolute magnetization of 1000 spins that
# start up, after a transient, should follow this curve, with a small finite-size tail of order `1 / sqrt(N)` above
# `T_c`.


def mean_field_magnetization(temperature):
    """Return the positive root of `m = tanh(m / T)`, or 0 above the critical temperature."""
    if temperature >= 1:
        return 0.0
    return brentq(lambda m: m - np.tanh(m / temperature), 1e-9, 1.0)


temperatures = np.linspace(0.4, 1.6, 9)
measured = []
for temperature in temperatures:
    ordered = create(MODEL, n_spins=1000, temperature=temperature, initial='up')
    run = Simulation(ordered, dt=1.0, random_state=0, observables=describe(MODEL).observables_of(ordered)).run(until=40)
    measured.append(np.abs(run.trajectory_['magnetization'][10:]).mean())
grid = np.linspace(0.3, 1.7, 200)
fig, (series, curve) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, simulation.trajectory_['magnetization'])
series.axhline(mean_field_magnetization(0.5), color='black', linestyle='--', label='root of m = tanh(2 m)')
series.set_xlabel('time')
series.set_ylabel('magnetization')
series.set_title('Ordering of 200 spins at T = 0.5 J')
series.legend()
curve.plot(grid, [mean_field_magnetization(value) for value in grid], color='black', label='mean field')
curve.plot(temperatures, measured, 'o', label='simulation, N = 1000')
curve.set_xlabel('temperature T / J')
curve.set_ylabel('mean |m|')
curve.set_title('Magnetization vanishes at T_c = J')
curve.legend()
fig.tight_layout()
plt.show()
