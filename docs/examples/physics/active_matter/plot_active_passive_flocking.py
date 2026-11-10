"""
# Flocking with active and passive agents

Forty-five active agents move among 297 passive ones in a dense periodic box, the setting of figure 10.7 of Charbonneau
(2017). Active agents keep a target speed and follow the velocities of the agents around them, while passive agents only
slow down. All agents repel each other when they overlap. The active agents gather into flocks that open channels
through the crowd, and the passive agents settle into still patches. The two kinds have different colours in the
primary view.
"""

# %%
# The agents start at uniform random positions, with velocity components up to 1. The radius `r_0 = 0.025`, the
# repulsion `epsilon = 25`, the flocking radius 0.1, the flocking amplitude 1, the target speed 0.02, the propulsion
# `mu = 10`, and the random force `eta = 0.1` are those of table 10.1, which give the compactness 0.672. Passive agents
# do not flock, as Silverberg et al. (2013) state and as the still passive agents of the figure require.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import ActivePassiveFlockingSystem

SYSTEM = 'active_passive_flocking'
SETTINGS = {'backend': NumbaBackend(), 'random_state': 0}


system = ActivePassiveFlockingSystem()
size = system.prepare().system_.space.extent[0]
box = rrb.Spatial2DView(
    origin='/',
    name='agents',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, size), y_range=(-size, 0.0)),
)
blueprint = rrb.Blueprint(
    box,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, backend=NumbaBackend(), observers=[observer], random_state=0)

# %%
# Snapshot and mean distances
# ---------------------------
# The left panel shows the agents at the end of the primary run. The right panel compares the mean distance within each
# kind of agent with and without the flocking force. From a uniform start both distances are near 0.38. With flocking
# on, the mean distance between the active agents should fall below the one between the passive agents, and with
# flocking off the two should stay together. The mean distance is a coarse measure in a periodic box, so the snapshot
# is the better guide to the flocks.

positions = trajectory.get_values('agents.x')[-1]
active = trajectory.get_values('agents.active')[-1] == 1.0
fig, (snapshot, distance) = plt.subplots(1, 2, figsize=(10, 4.5))
snapshot.scatter(*positions[~active].T, s=14, color='tab:green', label='passive')
snapshot.scatter(*positions[active].T, s=14, color='tab:red', label='active')
snapshot.set_xlim(0.0, size)
snapshot.set_ylim(0.0, size)
snapshot.set_aspect('equal')
snapshot.set_title('Agents after 50 time units')
snapshot.legend(loc='upper right')
without = simulate(ActivePassiveFlockingSystem(alpha=0.0), **SETTINGS)
distance.plot(trajectory.times, trajectory.get_values('active_distance'), color='tab:red', label='active, flocking on')
distance.plot(
    trajectory.times,
    trajectory.get_values('passive_distance'),
    color='tab:green',
    label='passive, flocking on',
)
distance.plot(
    without.times,
    without.get_values('active_distance'),
    color='tab:red',
    linestyle=':',
    label='active, flocking off',
)
distance.plot(
    without.times,
    without.get_values('passive_distance'),
    color='tab:green',
    linestyle=':',
    label='passive, flocking off',
)
distance.set_xlabel('time')
distance.set_ylabel('mean distance')
distance.set_title('Active agents gather only with the flocking force')
distance.legend()
fig.tight_layout()
plt.show()
