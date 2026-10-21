"""
# Swarmtaxis

A swarm of robots that only keep radio contact and avoid each other drifts as a whole towards a beacon that only the
robots on its side can see: the illuminated robots keep a wider berth, which pushes the swarm towards the light.
"""

# %%
# The model is created by name from the catalog with 20 robots starting in a cluster near the left edge of an arena of
# size 100, with the beacon at the middle of the right edge, and runs for 3000 steps. The blueprint shows the robots in
# the left half of the arena, drawn larger than the model's visual so they read in the thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'swarmtaxis'

arena = rrb.Spatial2DView(
    origin='/robots',
    name='arena',
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 60.0), y_range=(30.0, 70.0)),
    overrides={'robots': rr.Points2D.from_fields(radii=0.8).visualizer()},
)
blueprint = rrb.Blueprint(
    arena,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=15, observers=[observer]).run(until=3000)

# %%
# Drift towards the beacon
# ------------------------
# The centre of the swarm should move steadily towards the beacon at `x = 100`, while its spread stays bounded because
# the robots keep radio contact, and at any time only part of the swarm should see the beacon.

trajectory = simulation.trajectory_
frames = range(len(trajectory.times))
centres = np.array([trajectory.agents('robots', frame).locations.mean(axis=0) for frame in frames])
spreads = [trajectory.agents('robots', frame).locations.std(axis=0).mean() for frame in frames]
lit = [trajectory.agents('robots', frame).values['illuminated'].mean() for frame in frames]
fig, (drift, shares) = plt.subplots(1, 2, figsize=(10, 4))
drift.plot(trajectory.times, centres[:, 0], label='x of the centre')
drift.plot(trajectory.times, spreads, label='spread of the swarm')
drift.set_xlabel('step')
drift.set_ylabel('distance')
drift.set_title('The swarm moves towards the beacon at x = 100')
drift.legend()
shares.plot(trajectory.times, lit)
shares.set_ylim(0, 1)
shares.set_xlabel('step')
shares.set_ylabel('share of robots that see the beacon')
shares.set_title('Only part of the swarm is illuminated')
fig.tight_layout()
plt.show()
