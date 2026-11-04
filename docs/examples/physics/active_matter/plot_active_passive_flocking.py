"""
# Flocking with active and passive agents

Ninety active agents push through 210 passive ones in a dense periodic box. Active agents keep a preferred speed and
align with the mean velocity of the active agents around them, while passive agents only slow down. All agents repel
each other when they overlap. The two groups have different colours in the primary view.
"""

# %%
# The box has the packing fraction 0.94, and the repulsion `epsilon = 100`, the propulsion `mu = 0.05`, the speed
# `v0 = 1`, and the flocking radius 4 are those of Silverberg et al. (2013). The alignment strength is 2, and the noise
# amplitude of the active agents is 0.5.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe, run

MODEL = 'active_passive_flocking'

system = create(MODEL)
size = system.space.extent[0]
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
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = (
    Simulation(
        system,
        dt=entry.defaults['dt'],
        solver=entry.defaults['solver'],
        backend=entry.defaults['backend'],
        observers=[observer],
        random_state=1,
        observables=entry.observables_of(system),
    )
    .check()
    .prepare()
    .run(until=entry.defaults['until'])
)

# %%
# Polarization and mean distances
# -------------------------------
# With every agent active in a box of packing fraction 0.3, the polarization `|<v>| / <|v|>` should rise above 0.9 under
# alignment and stay near 0 without it. The right panel shows the mean distance between the active agents and between
# the passive agents of the primary run. They stay within 0.3 of each other, so this view shows no segregation of the
# groups. Silverberg et al. report segregation from random positions, but 300 agents with alignment 1 or 5 still had
# both mean distances near 12.1 after 3000 time units, far beyond the budget of this example.

SETTINGS = {'until': 60.0, 'dt': 1.0, 'random_state': 1}
fig, (order, distance) = plt.subplots(1, 2, figsize=(10, 4))
for alpha, label in ((1.0, 'alignment 1'), (0.0, 'no alignment')):
    found = run(MODEL, active_share=1.0, packing_fraction=0.3, alpha=alpha, noise=0.1, settings=SETTINGS).trajectory_
    order.plot(found.times, found.values('polarization'), label=label)
order.axhline(0.9, color='0.6', linestyle='--', label='0.9')
order.set_xlabel('time')
order.set_ylabel('polarization')
order.set_title('Alignment orders the velocities of active agents')
order.legend()
trajectory = simulation.trajectory_
distance.plot(trajectory.times, trajectory.values('active_distance'), label='active')
distance.plot(trajectory.times, trajectory.values('passive_distance'), label='passive')
distance.set_xlabel('time')
distance.set_ylabel('mean distance')
distance.set_ylim(0.0, 1.2 * trajectory.values('passive_distance').max())
distance.set_title('The mean distances within the groups')
distance.legend()
fig.tight_layout()
plt.show()
