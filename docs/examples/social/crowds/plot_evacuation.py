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

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ParticleKinetics
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'evacuation'

# The crowd's lattice of rooms and speeds is not a floor plan, so the primary view is the evacuated fraction.
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/evacuated', name='evacuated fraction', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL)
observables = describe(MODEL).observables_of(system)
simulation = Simulation(system, observers=[observer], dt=0.2, observables=observables).run(until=20)

# %%
# Particles against the gain and loss equations
# ---------------------------------------------
# The particle solver samples the same encounters with finite pedestrians. Its evacuated fraction should follow the
# curve of the gain and loss equations within the sampling error of a few thousand pedestrians, and a crowd that slows
# down more in encounters should take longer to leave.

times = simulation.trajectory_.times
sampled = Simulation(system, dt=0.2, observables=observables, solver=ParticleKinetics(n=5000), random_state=0)
particles = sampled.run(until=20).trajectory_['evacuated']
crowded_system = create(MODEL, crowding=8.0)
crowded = Simulation(crowded_system, dt=0.2, observables=describe(MODEL).observables_of(crowded_system)).run(until=20)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(times, simulation.trajectory_['evacuated'], label='gain and loss equations')
ax.plot(times, particles, linestyle='--', label='5000 particles')
ax.plot(times, crowded.trajectory_['evacuated'], label='equations, four times the crowding')
ax.set_xlabel('time')
ax.set_ylabel('evacuated fraction')
ax.set_title('Evacuation of a building of seven rooms')
ax.legend()
plt.show()
