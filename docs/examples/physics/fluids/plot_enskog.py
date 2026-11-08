"""
# Enskog dense gas

Disks of a finite size collide more often than point particles at the same density, because their neighbours crowd
around them. The local Enskog equation adds this to the Boltzmann equation by multiplying the collision frequency at
every position by Henderson's contact value `(1 - 7 c / 16) / (1 - c)^2` of the pair correlation, at the local area
fraction `c`. The chart shows a density wave along a periodic line in a dense gas, which fades, vanishes, and returns
as a sound wave.
"""

# %%
# The system is built from its class with disks of diameter 0.8 at the Knudsen number 0.2 of the dilute gas,
# 16 positions, and 16 points on each velocity axis. The stepper is the registered Strang splitting of semi-Lagrangian
# streaming and of collisions advanced by RK4. The observable `density_wave` is the component of wave number 1 of the
# density over position, computed at every frame of 0.1 up to `t = 5.5`, and the default blueprint shows the observables
# first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, SemiLagrangianSolver, StrangSplittingSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.physics.fluids import BoltzmannEquationSystem, EnskogSystem

SYSTEM = 'enskog'
POSITIONS, VELOCITIES = 16, 16
SPACING = 12.0 / VELOCITIES
KNUDSEN = 0.2


def density_wave(_, values):
    """Return the component of wave number 1 of the density over position."""
    density = np.reshape(values['gas.f'], (POSITIONS, -1)).sum(axis=1) * SPACING**2
    return 2 * density @ np.sin(2 * np.pi * np.arange(POSITIONS) / POSITIONS) / POSITIONS


observables = [ObservableSpec('density_wave', function=density_wave)]
system = EnskogSystem(positions=POSITIONS, velocities=VELOCITIES, knudsen=KNUDSEN, diameter=0.8)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='observables', interval=1)
solver = StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(0.05), 0.1)
trajectory = simulate(system, until=5.5, solver=solver, observers=[observer], observables=observables)

# %%
# Crowding acts as a smaller Knudsen number
# -----------------------------------------
# At a mean density of 1 the area fraction of disks of diameter `d` is `pi d^2 / 4`. A larger diameter should bring the
# gas closer to a fluid, in which the density wave returns strongly after it vanishes. If the contact value at the mean
# density is what matters, each curve should fall on the markers of the Boltzmann equation at the Knudsen number divided
# by that contact value.


def relative_wave(gas):
    """Return the times and the density wave of a gas, relative to its start."""
    run = simulate(gas, until=5.5, solver=solver, dt=0.1, observables=observables)
    wave = run.values('density_wave')
    return run.times, wave / wave[0]


fig, axis = plt.subplots(figsize=(7, 4))
for diameter in (0.01, 0.5, 0.8):
    fraction = np.pi * diameter**2 / 4
    contact = (1 - 7 * fraction / 16) / (1 - fraction) ** 2
    dense = EnskogSystem(positions=POSITIONS, velocities=VELOCITIES, knudsen=KNUDSEN, diameter=diameter)
    times, wave = relative_wave(dense.prepare().system_)
    line = axis.plot(times, wave, label=f'Enskog, diameter {diameter:g}, contact value {contact:.2f}')[0]
    dilute = BoltzmannEquationSystem(positions=POSITIONS, velocities=VELOCITIES, knudsen=KNUDSEN / contact)
    times, wave = relative_wave(dilute.prepare().system_)
    axis.plot(times[::4], wave[::4], 'o', markersize=4, color=line.get_color())
axis.plot([], [], 'o', markersize=4, color='gray', label='Boltzmann, Knudsen number over the contact value')
axis.set_xlabel('time t')
axis.set_ylabel('density wave relative to its start')
axis.set_title('Crowding speeds up collisions')
axis.legend()
fig.tight_layout()
plt.show()
