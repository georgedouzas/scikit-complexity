"""
# Carleman discrete-velocity gas

In Carleman's system of a gas the particles move only to the right or to the left at the speed `c`, and collisions
turn pairs of particles moving the same way into pairs moving opposite ways, at the rate
`kappa (f_minus^2 - f_plus^2)`. A pulse of right movers on a uniform background therefore splits: part of it runs
on to the right, collisions feed a pulse that runs to the left, and both fade as collisions even out the two
densities. The space-time image shows the density of right movers, one row per saved step with the earliest at the
top: the pulse runs to the right, spreads, and fades.
"""

# %%
# The system is built from its class on a periodic line of length 10 with 200 finite-volume cells, speed
# `c = 1`, and a slow collision rate `kappa = 0.2`, so the pulse travels far before it fades. The catalog registers
# the strong-stability-preserving Runge-Kutta stepper for this system, so it is passed explicitly. The observer saves
# every frame of 0.015 with the default blueprint, whose primary view is the space-time image of the right movers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.fluids import DiscreteVelocityBoltzmannSystem

SYSTEM = 'discrete_velocity_boltzmann'

system = DiscreteVelocityBoltzmannSystem(kappa=0.2)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = simulate(system, until=3, dt=0.015, observers=[observer])

# %%
# Conserved mass and relaxing current
# -----------------------------------
# Collisions swap particles between the two directions, so the total mass `sum(f_plus + f_minus)` should stay at its
# initial value to round-off, while the total current `c sum(f_plus - f_minus)`, which collisions do not conserve,
# should decay towards zero as the gas relaxes to the equilibrium `f_plus = f_minus`, over a longer run.

trajectory = simulate(DiscreteVelocityBoltzmannSystem(), until=10, dt=0.05)
right, left = trajectory.values('gas.f_plus'), trajectory.values('gas.f_minus')
mass = (right + left).sum(axis=1)
current = (right - left).sum(axis=1)
fig, (conserved, relaxing) = plt.subplots(1, 2, figsize=(10, 4))
conserved.semilogy(trajectory.times[1:], np.abs(mass[1:] - mass[0]) / mass[0] + 1e-17)
conserved.set_xlabel('time t')
conserved.set_ylabel('relative change of the mass')
conserved.set_title('Mass is conserved to round-off')
relaxing.semilogy(trajectory.times, np.abs(current) / np.abs(current[0]))
relaxing.set_xlabel('time t')
relaxing.set_ylabel('current relative to its start')
relaxing.set_title('Collisions relax the current')
fig.tight_layout()
plt.show()
