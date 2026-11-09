"""
# Crowd dynamics and evacuation

A crowd fills the seven rooms of a building whose rooms form a tree, with the exit in the root room. Pedestrians walk
to the next room on the way out at a rate proportional to their speed, slow down when they meet others in a room, and
speed up again by the exit signs. The crowd queues in the rooms near the exit, and the evacuated fraction grows along
an S-shaped curve to the whole crowd.
"""

# %%
# The crowd is a distribution over pairs of a room and a speed class, a site group on a lattice indexed by room and
# speed. The blueprint shows the evacuated fraction, which rises from zero to one.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import KineticSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.crowds import EvacuationSystem

SYSTEM = 'evacuation'

# The crowd's lattice of rooms and speeds is not a floor plan, so the primary view is the evacuated fraction.
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/evacuated', name='evacuated fraction', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
system = EvacuationSystem()
trajectory = simulate(system, until=20, dt=0.2, observers=[observer])

# %%
# Particles against the gain and loss equations
# ---------------------------------------------
# The particle solver samples the same encounters with finite pedestrians. Its evacuated fraction should follow the
# curve of the gain and loss equations within the sampling error of a few thousand pedestrians, and a crowd that slows
# down more in encounters should take longer to leave.

times = trajectory.times
particles = simulate(system, until=20, dt=0.2, solver=KineticSolver(n=5000), random_state=0).get_values('evacuated')
crowded = simulate(EvacuationSystem(crowding=8.0), until=20, dt=0.2)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(times, trajectory.get_values('evacuated'), label='gain and loss equations')
ax.plot(times, particles, linestyle='--', label='5000 particles')
ax.plot(times, crowded.get_values('evacuated'), label='equations, four times the crowding')
ax.set_xlabel('time')
ax.set_ylabel('evacuated fraction')
ax.set_title('Evacuation of a building of seven rooms')
ax.legend()
plt.show()
