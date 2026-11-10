"""
# BGK shock tube

The Bhatnagar-Gross-Krook equation replaces binary collisions by a relaxation towards the Maxwellian with the same
density, momentum, and energy. A gas starts from the two states of the Sod shock tube, on either side of the middle of
a unit interval, and a rarefaction, a contact discontinuity, and a shock emerge. The image shows the density over
position and velocity, and the chart shows the density over position.
"""

# %%
# The example builds `BGKEquationSystem` with its default Knudsen number of 0.01, 100 positions, and 32
# velocities. The registered stepper splits the streaming, advanced by a strong-stability-preserving Runge-Kutta method,
# from the relaxation, and the observer saves every frame of 0.005 up to `t = 0.2`, with the default blueprint.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver, SSPRKSolver, StrangSplittingSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.fluids import BGKEquationSystem

SYSTEM = 'bgk_equation'

system = BGKEquationSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = StrangSplittingSolver(SSPRKSolver(), RK4Solver(0.005), 0.005)
simulation = simulate(system, solver=solver, dt=0.005, observers=[observer])

# %%
# The fluid limit
# ---------------
# As the Knudsen number falls, collisions keep the gas closer to a local Maxwellian, so the density moment should tend
# to the exact solution of the Euler equations with the adiabatic index 3 of one velocity dimension, with the sharp
# edges of its contact and its shock, though the finite-volume streaming still rounds them. At a large Knudsen number
# the gas barely collides, and free streaming alone should smear the two states into a ramp, with small steps from the
# discrete velocities.


def euler_density(points, time, gamma):
    """Return the density of the exact Euler solution of the Sod shock tube, with the discontinuity at zero."""
    left, right = (1.0, 1.0), (0.125, 0.1)
    sound_left = np.sqrt(gamma * left[1] / left[0])
    ratio = (gamma - 1) / (gamma + 1)

    def speed_mismatch(pressure):
        fan = 2 * sound_left / (gamma - 1) * ((pressure / left[1]) ** ((gamma - 1) / (2 * gamma)) - 1)
        return fan + (pressure - right[1]) * np.sqrt((1 - ratio) / right[0] / (pressure + ratio * right[1]))

    star = brentq(speed_mismatch, 1e-6, 1.0)
    speed = 2 * sound_left / (gamma - 1) * (1 - (star / left[1]) ** ((gamma - 1) / (2 * gamma)))
    density_left = left[0] * (star / left[1]) ** (1 / gamma)
    density_right = right[0] * (star / right[1] + ratio) / (star / right[1] * ratio + 1)
    shock = density_right * speed / (density_right - right[0])
    tail = speed - np.sqrt(gamma * star / density_left)
    rays = points / time
    fan = left[0] * (2 / (gamma + 1) - ratio / sound_left * rays) ** (2 / (gamma - 1))
    return np.select(
        [rays < -sound_left, rays < tail, rays < speed, rays < shock],
        [left[0], fan, density_left, density_right],
        right[0],
    )


positions, velocities = 200, 48
centres = (np.arange(positions) + 0.5) / positions
fig, axis = plt.subplots(figsize=(7, 4))
for knudsen, step in ((1.0, 2e-3), (1e-2, 2e-3), (1e-4, 2e-4)):
    fine = BGKEquationSystem(positions=positions, velocities=velocities, knudsen=knudsen)
    stepper = StrangSplittingSolver(SSPRKSolver(dt=0.001), RK4Solver(dt=step), dt=0.001)
    trajectory = simulate(fine, solver=stepper, dt=0.2)
    density = trajectory.get_values('gas.x')[-1].reshape(positions, velocities).sum(axis=1) * 12.0 / velocities
    axis.plot(centres, density, label=f'Knudsen {knudsen:g}')
axis.plot(centres, euler_density(centres - 0.5, 0.2, gamma=3.0), 'k--', label='exact Euler solution')
axis.set_xlabel('position')
axis.set_ylabel('density at t = 0.2')
axis.set_title('The density tends to the Euler solution as collisions speed up')
axis.legend()
fig.tight_layout()
plt.show()
