"""
# Cucker-Smale flocking

Every bird steers its velocity towards the velocities of all the others, with weights `K / (1 + r^2)^beta` that
fade with the distance `r`. Started with random positions and velocities, the birds first scatter, then their
velocities converge to the common mean velocity, which is conserved, and the flock moves on as a rigid formation.
The view shows 50 birds, which end up crossing the plane together.
"""

# %%
# The model is created by name from the catalog with coupling `K = 1` and decay exponent `beta = 0.25`. The fourth-order
# Runge-Kutta stepper takes steps of 0.05 and the observer saves every frame of 0.1 with the default blueprint, whose
# primary view shows the birds.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cucker_smale_flocking'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = RK4(dt=0.05)
simulation = Simulation(system, solver=solver, dt=0.1, observers=[observer], random_state=0).run(until=20)

# %%
# Unconditional flocking
# ----------------------
# Cucker and Smale proved that for `beta < 1/2` the spread of the velocities about their mean decays to zero for every
# initial condition, exponentially once the flock has a bounded diameter, while the mean velocity stays constant. For
# larger `beta` the weights fade fast and a flock that starts spread out can break up. The velocity spread should
# fall exponentially for `beta = 0.25`, and much more slowly, or not to zero, for `beta = 1.5`.

fig, ax = plt.subplots(figsize=(7, 4))
for beta, color in ((0.25, 'tab:blue'), (1.5, 'tab:orange')):
    run = Simulation(create(MODEL, beta=beta, random_state=1), solver=solver, dt=0.5, random_state=1).run(until=60)
    velocities = run.trajectory_['birds.v']
    spread = np.sqrt(np.mean(np.sum((velocities - velocities.mean(axis=1, keepdims=True)) ** 2, axis=-1), axis=1))
    ax.semilogy(run.trajectory_.times, spread, color=color, label=f'beta = {beta}')
ax.set_xlabel('time t')
ax.set_ylabel('velocity spread about the mean')
ax.set_title('Velocities align for beta < 1/2')
ax.legend()
fig.tight_layout()
plt.show()
