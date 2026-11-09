"""
# Optical trap

A focused laser beam pulls a colloidal bead towards its focus with a nearly harmonic force, while thermal kicks push it
around. A trap that is four times stiffer along one axis than along the other holds the beads in an elongated cloud,
wider along the soft axis. The view shows 500 independent beads released at the focus, filling the elliptical
thermal cloud of the trap.
"""

# %%
# The system is built from its class with stiffnesses `k = (1, 0.25)`, friction `gamma = 1`, and thermal
# energy `kT = 1`. The Euler-Maruyama stepper takes steps of 0.01 and the observer saves every frame of 0.1 with the
# default blueprint, whose primary view shows the beads.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import distribution
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.soft_matter import OpticalTrapSystem

SYSTEM = 'optical_trap'

system = OpticalTrapSystem(n_particles=500, k=(1.0, 0.25))
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = EulerMaruyamaSolver(dt=0.01)
trajectory = simulate(system, until=25, solver=solver, observers=[observer], random_state=0)

# %%
# Equipartition calibrates the trap
# ---------------------------------
# In equilibrium each coordinate is Gaussian with variance `kT / k`, the equipartition theorem that experiments use
# to calibrate the stiffness of optical tweezers. Pooled over the beads and the frames after the relaxation time
# `gamma / k = 4` of the soft axis, the positions along each axis should follow these Gaussians, twice as wide along
# the soft axis.

positions = trajectory.get_values('beads.x')[150:]
fig, ax = plt.subplots(figsize=(7, 4))
for axis, (stiffness, color) in enumerate(((1.0, 'tab:blue'), (0.25, 'tab:orange'))):
    values = positions[..., axis].ravel()
    histogram = distribution(values, kind='histogram', bins=40)
    edges, density = histogram.edges, histogram.density
    grid = np.linspace(edges[0], edges[-1], 200)
    ax.stairs(density, edges, color=color, label=f'axis {axis}, measured k = {1 / values.var():.2f}')
    gaussian = np.sqrt(stiffness / (2 * np.pi)) * np.exp(-stiffness * grid**2 / 2)
    ax.plot(grid, gaussian, color=color, linestyle='--', label=f'Gaussian of variance kT / {stiffness}')
ax.set_xlabel('position')
ax.set_ylabel('probability density')
ax.set_title('Positions in a trap of stiffnesses 1 and 0.25')
ax.legend()
fig.tight_layout()
plt.show()
