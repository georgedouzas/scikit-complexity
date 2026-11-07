"""
# Schwarzschild precession

A planet on a bound orbit around a static mass does not retrace its ellipse: in general relativity its closest
approach advances a little every turn, so the orbit draws a rosette.
"""

# %%
# The planet follows a timelike geodesic of the Schwarzschild metric, whose Christoffel symbols the space derives from
# the symbolic metric. Units have `G = c = 1` and the mass `M = 1`. The orbit has the semi-major axis `a = 100` and the
# eccentricity `e = 0.5`, and the run covers about three turns of its proper time. The default blueprint draws the
# planet in the equatorial plane with its path.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.celestial_mechanics import SchwarzschildPrecessionSystem

SYSTEM = 'schwarzschild_precession'

system = SchwarzschildPrecessionSystem(semi_major_axis=100.0, eccentricity=0.5)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-10)
trajectory = simulate(system, solver=solver, observers=[observer])

# %%
# Advance of the periapsis
# ------------------------
# Far from the horizon the periapsis advances by `6 π M / (a (1 - e²))` per turn, about 0.25 radians here. The angles
# where the radial velocity turns positive, less whole turns, should step by about that much.

x = np.asarray(trajectory.values('particle.x'))
u = np.asarray(trajectory.values('particle.x_d1'))
radius, angle = x[:, 1], x[:, 3]
index = np.flatnonzero((u[:-1, 1] < 0) & (u[1:, 1] >= 0))
advances = np.diff(np.concatenate([[0.0], angle[index]])) - 2 * np.pi
fig, (orbit, steps) = plt.subplots(1, 2, figsize=(10, 4))
orbit.plot(radius * np.cos(angle), radius * np.sin(angle))
orbit.plot(0, 0, '*', color='black', markersize=10)
orbit.set_aspect('equal')
orbit.set_title('A rosette instead of an ellipse')
steps.bar(np.arange(1, len(advances) + 1), advances, label='simulated')
steps.axhline(6 * np.pi / (100 * 0.75), color='black', linestyle='--', label='6 π M / (a (1 - e²))')
steps.set_xlabel('turn')
steps.set_ylabel('advance of the periapsis (rad)')
steps.legend()
fig.tight_layout()
plt.show()
