"""
# Nonreciprocal Vicsek system

Two species of movers align with their own kind, and species A aligns with species B while B antialigns with A. Neither
can settle, so the two flocks chase each other: their mean headings stay a quarter turn apart and turn together, the
chiral phase of nonreciprocal matter. The points are coloured by species.
"""

# %%
# Four hundred movers, half of each species, move in a periodic box of side 5 at density 16, with `J_ab = 1`,
# `J_ba = -1`, and heading noise 0.3. The blueprint draws the movers larger than the system's visual does, so they read
# in the gallery thumbnail.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import NonreciprocalVicsekSystem

SYSTEM = 'nonreciprocal_vicsek'

box = rrb.Spatial2DView(
    origin='/movers',
    name='movers',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 5.0), y_range=(-5.0, 0.0)),
    overrides={'movers': rr.Points2D.from_fields(radii=0.06).visualizer()},
)
blueprint = rrb.Blueprint(
    box,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = NonreciprocalVicsekSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, until=250, backend=NumbaBackend(), observers=[observer], random_state=0)

# %%
# Chasing against flocking
# ------------------------
# With reciprocal alignment, `J_ab = J_ba = 1`, both species order in one common direction, which then stays put. With
# `J_ab = -J_ba`, the angle between the mean headings of the species should settle near a quarter turn, `pi / 2`, and
# the heading of species A should turn at a steady rate instead of settling, the signature of the chiral phase.


def headings(coupling):
    """Return the times and the mean headings of both species for a coupling `(J_ab, J_ba)`."""
    system_system = NonreciprocalVicsekSystem(J_ab=coupling[0], J_ba=coupling[1])
    run = simulate(system_system, backend=NumbaBackend(), random_state=0)
    return run.times, run.values('heading_a'), run.values('heading_b')


fig, (relative, turning) = plt.subplots(1, 2, figsize=(10, 4))
for coupling, label in (((1.0, 1.0), 'reciprocal'), ((1.0, -1.0), 'nonreciprocal')):
    times, heading_a, heading_b = headings(coupling)
    relative.plot(times, np.abs(np.angle(heading_a * np.conj(heading_b))), label=label)
    turning.plot(times, np.unwrap(np.angle(heading_a)), label=label)
relative.axhline(np.pi / 2, color='black', linestyle='--', label='quarter turn')
relative.set_xlabel('step')
relative.set_ylabel('|angle between the mean headings of A and B|')
relative.set_title('Angle between the species')
relative.legend()
turning.set_xlabel('step')
turning.set_ylabel('unwrapped mean heading of A')
turning.set_title('Turning of species A')
turning.legend()
fig.tight_layout()
plt.show()
