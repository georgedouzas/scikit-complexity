"""
# Hydrodynamically coupled traps

Two colloidal spheres held by optical tweezers a few micrometres apart are released together and caught again, over
and over. While they are free, the force of every thermal kick on one sphere moves the fluid around the other, so the
two diffuse together. The view shows the two spheres of radius 1 um, 3 um apart, as their traps blink.
"""

# %%
# The system is built from its class with traps of stiffness 0.2 pN/um that stay on for 0.3 s and off for
# 0.1 s, and spheres of radius 1 um in water at room temperature. Every step of 5 ms is an Ermak-McCammon step of
# Brownian dynamics, whose random displacements have the covariance `2 kT M dt` of the Rotne-Prager-Yamakawa mobility
# `M`, so the system runs in discrete time and its frames count steps. The observer saves every tenth step with the
# default blueprint.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.soft_matter import HydrodynamicallyCoupledTrapsSystem

SYSTEM = 'hydrodynamically_coupled_traps'
DT, ON, OFF = 5e-3, 0.3, 0.1

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
simulation = simulate(HydrodynamicallyCoupledTrapsSystem(), until=2000, observers=[observer], random_state=0)

# %%
# Released and caught again
# -------------------------
# While the traps are on, each sphere fluctuates about its centre with the variance `kT / k`. When they switch off the
# spheres wander away, and when they switch on again the traps pull them back within `gamma / k = 0.09 s`.

times = simulation.times * DT
positions = simulation.get_values('spheres.x')
fig, ax = plt.subplots(figsize=(7, 3.5))
for start in np.arange(ON, times[-1], ON + OFF):
    ax.axvspan(start, start + OFF, color='0.9')
ax.plot(times, positions[:, 0, 0] + 1.5, label='left sphere')
ax.plot(times, positions[:, 1, 0] - 1.5, label='right sphere')
ax.set_xlim(0, 4)
ax.set_xlabel('time (s)')
ax.set_ylabel('displacement along x (um)')
ax.set_title('Traps off in the grey windows')
ax.legend()
fig.tight_layout()
plt.show()

# %%
# The correlation of free motion follows the mobility
# ---------------------------------------------------
# Batchelor showed that the steps of two free spheres correlate as the ratio `M_12 / M_11` of their mobilities. The
# Rotne-Prager-Yamakawa tensor gives `3 a / (2 r) - a^3 / r^3` along the line of centres and `3 a / (4 r) + a^3 /
# (2 r^3)` across it, and the Oseen tensor its leading terms. The steps of the spheres while the traps are off should
# follow these curves at every separation, and should not correlate at all without the hydrodynamic coupling.

separations = np.array([2.2, 2.5, 3.0, 4.0, 6.0, 8.0])


def free_correlations(separation, tensor, seed):
    """Return the correlations of the free steps of the spheres along and across their line of centres."""
    system = HydrodynamicallyCoupledTrapsSystem(separation=separation, tensor=tensor)
    trajectory = simulate(system, until=10000, random_state=seed)
    free = (trajectory.times[:-1] * DT) % (ON + OFF) >= ON - 1e-9
    steps = np.diff(trajectory.get_values('spheres.x'), axis=0)[free]
    return [np.corrcoef(steps[:, 0, axis], steps[:, 1, axis])[0, 1] for axis in (0, 1)]


coupled = np.array([free_correlations(r, 'rotne-prager-yamakawa', seed) for seed, r in enumerate(separations)])
uncoupled = np.array([free_correlations(r, 'free-draining', seed) for seed, r in enumerate(separations)])
grid = np.linspace(2.0, 8.5, 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid, 1.5 / grid - 1 / grid**3, color='tab:blue', label='RPY, along')
ax.plot(grid, 0.75 / grid + 0.5 / grid**3, color='tab:orange', label='RPY, across')
ax.plot(grid, 1.5 / grid, color='tab:blue', linestyle=':', label='Oseen, along')
ax.plot(grid, 0.75 / grid, color='tab:orange', linestyle=':', label='Oseen, across')
ax.plot(separations, coupled[:, 0], 'o', color='tab:blue', label='simulation, along')
ax.plot(separations, coupled[:, 1], 's', color='tab:orange', label='simulation, across')
ax.plot(separations, uncoupled[:, 0], 'x', color='tab:gray', label='free-draining, along')
ax.axhline(0.0, color='black', linewidth=0.5)
ax.set_xlabel(f'distance between the centres (um), radius {1.0:.0f} um')
ax.set_ylabel('correlation of the free steps')
ax.set_title('Hydrodynamic correlation of two released spheres')
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
plt.show()
