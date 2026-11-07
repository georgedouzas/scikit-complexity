"""
# Barkley

Orthogonal stripes of excitation and recovery in Barkley's excitable medium curl into a rotating spiral wave. With a
steeper recovery the spiral is unstable and breaks into many short waves, the spatiotemporal chaos of fibrillation.
"""

# %%
# The system is created by name from the catalog on a lattice of 64 by 64 sites spaced by 3, in the regime of a stable
# spiral, and integrated with Runge-Kutta steps of 0.2. The observer saves every frame with the default blueprint,
# whose primary view is the image of the excitation.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'barkley'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=4.0, solver=RK4Solver(dt=0.2), observers=[observer]).prepare().run(until=400)

# %%
# Spiral and breakup
# ------------------
# The stable regime should keep one rotating spiral, while the regime `(a, epsilon, gamma) = (0.7, 0.08, 3)` should
# break the wave into many fragments that keep a large part of the medium excited.

chaotic = create_system(SYSTEM, a=0.7, epsilon=0.08, gamma=3.0)
chaos = Simulation(chaotic, dt=4.0, solver=RK4Solver(dt=0.2)).prepare().run(until=400).trajectory_
fig, (spiral, broken) = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, trajectory, title in ((spiral, simulation.trajectory_, 'gamma = 1'), (broken, chaos, 'gamma = 3')):
    image = ax.imshow(trajectory.lattice('medium.u')[-1], cmap='inferno', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Excitation at t = 400, {title}')
    ax.set_xlabel('column')
    ax.set_ylabel('row')
fig.colorbar(image, ax=[spiral, broken], label='excitation u')
plt.show()
