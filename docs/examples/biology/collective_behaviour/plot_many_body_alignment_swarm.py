"""
# Many-body alignment swarm

Forty self-propelled agents align with their neighbours through pairs, triples, and quadruples of them. Besides a
Morse force and pairwise alignment, an agent flanked by two neighbours on opposite sides aligns with their mean
velocity, and an agent surrounded by three neighbours aligns with all three, each term of exactly one body order.
"""

# %%
# The system is built from its class with forty agents near a lattice in a periodic box, at the preferred speed with
# random headings. Its defaults switch on the 3-body flanking term, `kappa_v = 2`, and the 4-body encirclement term,
# `kappa_4 = 4`.
# The observer saves every frame with the default blueprint, whose primary view shows the agents in the plane.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import alignment
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.collective_behaviour import ManyBodyAlignmentSwarmSystem

SYSTEM = 'many_body_alignment_swarm'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(
    ManyBodyAlignmentSwarmSystem(40, random_state=0),
    observers=[observer],
    until=100,
    backend=NumbaBackend(),
)

# %%
# Body order by body order
# ------------------------
# The same agents run as the pairwise collective, `kappa_v = kappa_4 = 0`, and as the 3-body collective,
# `kappa_4 = 0`. The polar order, the length of the mean heading, measures how far each collective aligns. All three
# should align into one flock. The many-body terms act only on agents with two or three neighbours around them, so they
# change the path to the flock rather than its end.

pairwise = ManyBodyAlignmentSwarmSystem(40, kappa_v=0.0, kappa_4=0.0, random_state=0)
flanking = ManyBodyAlignmentSwarmSystem(40, kappa_4=0.0, random_state=0)
collectives = {
    'pairwise': simulate(pairwise, until=100, backend=NumbaBackend()),
    '3-body': simulate(flanking, until=100, backend=NumbaBackend()),
    '2+3+4-body': trajectory,
}
fig, (order, path) = plt.subplots(1, 2, figsize=(11, 4))
for name, run in collectives.items():
    order.plot(run.times, alignment(run, kind='mean-vector').order, label=name)
order.set_ylim(0, 1.05)
order.set_xlabel('time')
order.set_ylabel('polar order')
order.set_title('Alignment of the three collectives')
order.legend()
positions, velocities = trajectory.values('agents.x')[-1], trajectory.values('agents.x_d1')[-1]
path.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1])
path.set_aspect('equal')
path.set_xlabel('x')
path.set_ylabel('y')
path.set_title('Headings of the 2+3+4-body swarm at the end')
fig.tight_layout()
plt.show()

# %%
# Speed
# -----
# The self-propulsion holds every agent near the speed `sqrt(alpha / beta) = 1` in every collective.

speeds = {name: np.linalg.norm(run.values('agents.x_d1'), axis=2) for name, run in collectives.items()}
{name: (round(float(value.min()), 2), round(float(value.max()), 2)) for name, value in speeds.items()}
