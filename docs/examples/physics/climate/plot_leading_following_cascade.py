"""
# Leading-following tipping cascade

A leading tipping element is pushed through its fold by a slowly rising control, and its state forces a following
element. A strong enough coupling makes the follower tip too, a domino cascade, and a follower near a Hopf bifurcation
starts to oscillate instead.
"""

# %%
# The control of the leader rises from -1 to 1 at the rate 0.004, past the fold at `2 / (3 √3)`. The default blueprint
# shows the leader and the fold follower, coupled with the strength 0.4.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'leading_following_cascade'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.5, solver=EulerMaruyamaSolver(dt=0.01), observers=[observer], random_state=0)
simulation.prepare().run(until=500)

# %%
# Domino cascade or not
# ---------------------
# The follower tips only when the tipped leader shifts its control past its own fold, for a coupling above about
# 0.29. A Hopf follower should instead leave its focus and settle on a cycle once the leader has tipped.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for coupling in (0.2, 0.4):
    run = Simulation(
        create_system(SYSTEM, coupling=coupling),
        dt=0.5,
        solver=EulerMaruyamaSolver(dt=0.01),
        random_state=0,
    )
    trajectory = run.prepare().run(until=500).trajectory_
    times = np.asarray(trajectory.times)
    axes[0].plot(times, np.ravel(trajectory.values('follower.y')), label=f'follower, coupling = {coupling}')
axes[0].plot(times, np.ravel(trajectory.values('leader.x')), '--', color='black', label='leader')
axes[0].set_xlabel('time')
axes[0].set_title('Fold follower')
axes[0].legend()
run = Simulation(create_system(SYSTEM, follower='hopf'), dt=0.5, solver=EulerMaruyamaSolver(dt=0.01), random_state=0)
trajectory = run.prepare().run(until=500).trajectory_
axes[1].plot(np.asarray(trajectory.times), np.ravel(trajectory.values('follower.y')), label='Hopf follower')
axes[1].plot(np.asarray(trajectory.times), np.ravel(trajectory.values('leader.x')), '--', color='black', label='leader')
axes[1].set_xlabel('time')
axes[1].set_title('Hopf follower')
axes[1].legend()
plt.show()
