"""
# Boltzmann gas of Maxwell molecules

The Boltzmann equation follows the density of a gas over position and velocity, with free streaming along position and
binary collisions at every position that redistribute the density over velocity while conserving mass, momentum, and
energy. A density wave along a periodic line starts on a Maxwellian of unit temperature, and the Knudsen number, the
inverse of the collision frequency, decides its fate. The chart shows the wave at a small Knudsen number, which fades,
vanishes, and returns as a sound wave.
"""

# %%
# The system is built from its class with 16 positions, 16 points on each of the two velocity axes, and the
# Knudsen number 0.02. The catalog registers a Strang splitting of semi-Lagrangian streaming and of collisions by the
# fast spectral method, advanced by RK4. The observable `density_wave` is the component of wave number 1 of the density
# over position, computed at every frame of 0.1 up to `t = 5.5`, and the default blueprint shows the observables first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, SemiLagrangianSolver, StrangSplittingSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.physics.fluids import BoltzmannEquationSystem

SYSTEM = 'boltzmann_equation'
POSITIONS, VELOCITIES = 16, 16
SPACING = 12.0 / VELOCITIES


def density_wave(_, values):
    """Return the component of wave number 1 of the density over position."""
    density = np.reshape(values['gas.x'], (POSITIONS, -1)).sum(axis=1) * SPACING**2
    return 2 * density @ np.sin(2 * np.pi * np.arange(POSITIONS) / POSITIONS) / POSITIONS


def solver_of(step):
    """Return the Strang splitting of a gas, with a collision step short enough for its Knudsen number."""
    return StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(step), 0.1)


observables = [ObservableSpec('density_wave', function=density_wave)]
system = BoltzmannEquationSystem(positions=POSITIONS, velocities=VELOCITIES, knudsen=0.02)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='observables', interval=1)
trajectory = simulate(system, until=5.5, solver=solver_of(0.02), observers=[observer], observables=observables)

# %%
# From free streaming to sound
# ----------------------------
# Without collisions, particles of different speeds run apart and the wave should fade as `exp(-t^2 / 2)`. With fast
# collisions, the gas behaves as a fluid: a density wave at uniform temperature is half a sound wave of speed `sqrt(2)`,
# the adiabatic speed of a gas with two velocity dimensions, and half a standing temperature wave, so it should follow
# `(1 + cos(sqrt(2) t)) / 2`, vanish near `t = 2.2`, and return near `t = 4.4`, damped by viscosity and heat conduction.
# In between, the wave should barely return, more weakly the larger the Knudsen number.

times = np.linspace(0, 5.5, 100)
fig, axis = plt.subplots(figsize=(7, 4))
for knudsen, step in ((1000.0, 0.05), (1.0, 0.05), (0.2, 0.05), (0.02, 0.02)):
    gas = BoltzmannEquationSystem(positions=POSITIONS, velocities=VELOCITIES, knudsen=knudsen)
    run = simulate(gas, until=5.5, solver=solver_of(step), observables=observables)
    wave = run.values('density_wave')
    axis.plot(run.times, wave / wave[0], label=f'Knudsen {knudsen:g}')
axis.plot(times, np.exp(-(times**2) / 2), 'k--', label='free streaming')
axis.plot(times, (1 + np.cos(np.sqrt(2) * times)) / 2, 'k:', label='Euler fluid')
axis.set_xlabel('time t')
axis.set_ylabel('density wave relative to its start')
axis.set_title('Collisions turn a fading wave into sound')
axis.legend(loc='upper center')
fig.tight_layout()
plt.show()
