"""
# Brownian particles with energy depots

Schweitzer and Ebeling's active particles take up energy from their surroundings into an internal depot and convert
it into motion, against friction. When the intake is strong enough the resting state is unstable: particles that start
almost at rest speed up until intake and friction balance, then keep a constant speed in a slowly wandering
direction. The view shows 50 particles started at the origin, flying apart along nearly straight paths.
"""

# %%
# The model is created by name from the catalog with friction `gamma = 20`, intake `q = 10`, internal loss `c = 1`,
# conversion `d = 10`, and velocity noise `D = 0.01`. The Euler-Maruyama stepper takes steps of 0.01 and the observer
# saves every frame of 0.1 with the default blueprint, whose primary view shows the particles.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'depot_brownian_particles'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.01)
simulation = Simulation(system, solver=solver, dt=0.1, observers=[observer], random_state=0).run(until=20)

# %%
# Bifurcation to active motion
# ----------------------------
# The stationary speed of the deterministic dynamics is `v0 = sqrt(q / gamma - c / d)` when the intake exceeds
# `q_c = gamma c / d = 2`, and zero below it, a pitchfork bifurcation from rest to motion. The mean speed of the
# particles at the end of runs over a range of intakes should follow this curve, rounded slightly by the noise near
# `q_c`.

intakes = np.linspace(0.5, 12, 12)
speeds = []
for intake in intakes:
    run = Simulation(create(MODEL, q=intake, random_state=1), solver=solver, dt=0.5, random_state=1).run(until=40)
    speeds.append(np.linalg.norm(run.trajectory_['walkers.v'][-20:], axis=-1).mean())
grid = np.linspace(0.5, 12, 300)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(intakes, speeds, 'o', label='simulation, 50 particles')
ax.plot(grid, np.sqrt(np.clip(grid / 20 - 0.1, 0, None)), color='black', label='v0 = sqrt(q / gamma - c / d)')
ax.axvline(2.0, color='gray', linestyle='--', label='q_c = gamma c / d')
ax.set_xlabel('energy intake q')
ax.set_ylabel('mean speed')
ax.set_title('Particles start moving above the critical intake')
ax.legend()
fig.tight_layout()
plt.show()
