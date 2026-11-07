"""
# Oppenheimer-Snyder collapse

A ball of dust with no pressure to hold it up collapses under its own gravity: its surface falls through its
Schwarzschild radius, and the ball becomes a black hole.
"""

# %%
# The ball is a field of comoving shells on a radial lattice, whose areal radius obeys the radial Einstein equation of
# the Lemaître-Tolman-Bondi metric. The ball of mass `M = 1` and radius 10 starts at rest with uniform density, in
# units with `G = c = 1`, and the run stops just before it reaches zero size. The default blueprint shows the radius of
# every shell.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.cosmology import OppenheimerSnyderCollapseSystem

SYSTEM = 'oppenheimer_snyder_collapse'

system = OppenheimerSnyderCollapseSystem(n_shells=16, radius=10.0, mass=1.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)
trajectory = simulate(system, solver=solver, observers=[observer])

# %%
# The cycloid of every shell
# --------------------------
# A uniform ball stays uniform: every shell keeps `R = r a(t)` with the cycloid `a = (1 + cos η) / 2`,
# `t = (η + sin η) / (2 √k)` and `k = 2 M / R₀³`. The surface crosses `2 M`, where light can no longer leave it.

times = np.asarray(trajectory.times)
radii = np.asarray(trajectory.values('ball.R'))
k = 2 / 10.0**3
eta = np.linspace(0, 0.99 * np.pi, 400)
fig, (shells, ratio) = plt.subplots(1, 2, figsize=(10, 4))
shells.plot(times, radii[:, ::3])
shells.axhline(2.0, color='black', linestyle='--', label='Schwarzschild radius 2 M')
shells.set_xlabel('proper time')
shells.set_ylabel('areal radius R')
shells.set_title('Every shell falls in')
shells.legend()
ratio.plot(times, radii[:, -1] / 10.0, label='surface R / R₀')
ratio.plot((eta + np.sin(eta)) / (2 * np.sqrt(k)), (1 + np.cos(eta)) / 2, '--', color='black', label='cycloid')
ratio.set_xlabel('proper time')
ratio.set_title('Oppenheimer-Snyder cycloid')
ratio.legend()
fig.tight_layout()
plt.show()
