"""
# Echo state network

Two copies of a reservoir of 200 leaky tanh units, started from different random activities and driven by the same
sine, forget where they started: within a few dozen steps the activity of every unit becomes the same in both copies,
a function of the input history alone. This echo state property is what lets a trained readout use the reservoir.
"""

# %%
# The recurrent weights are sparse and Gaussian with spectral radius 0.9, and every unit leaks with rate 0.3. The
# primary view shows three units in both copies, one colour per unit, whose two lines merge as the copies forget
# their initial activities.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'echo_state_network'
UNITS = (0, 2, 4)
COLORS = ((31, 119, 180), (255, 127, 14), (44, 160, 44))

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/copies', name='three units in two copies', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, n_units=200, spectral_radius=0.9, leak=0.3)
first = Simulation(system, dt=1, observers=[observer]).run(until=100).trajectory_
start = np.random.default_rng(1).uniform(-1.0, 1.0, 200)
second = Simulation(system, dt=1, initial_state={'units.r': start}).run(until=100).trajectory_
time = [rr.TimeColumn('sim_time', duration=first.times)]
for copy, trajectory in (('first', first), ('second', second)):
    for unit, color in zip(UNITS, COLORS, strict=True):
        observer.recording_.log(
            f'copies/{copy}/unit_{unit}',
            rr.SeriesLines(colors=[color], names=f'unit {unit}, {copy} copy'),
            static=True,
        )
        observer.recording_.send_columns(
            f'copies/{copy}/unit_{unit}',
            indexes=time,
            columns=rr.Scalars.columns(scalars=trajectory['units.r'][:, unit]),
        )
observer.recording_.flush()

# %%
# Forgetting the initial state
# ----------------------------
# The distance between the activities of two copies driven by the same input should fall exponentially, by orders of
# magnitude per hundred steps, while the spectral radius stays below 1, and slower as it nears 1. With a spectral
# radius well above 1 the reservoir keeps dynamics of its own, and the distance need not fall.

fig, (activity, distance) = plt.subplots(1, 2, figsize=(10, 4))
for unit in UNITS:
    (line,) = activity.plot(first.times, first['units.r'][:, unit], label=f'unit {unit}')
    activity.plot(second.times, second['units.r'][:, unit], '--', color=line.get_color())
activity.set_xlabel('step t')
activity.set_ylabel('activity r')
activity.set_title('Two copies, solid and dashed')
activity.legend()
for radius in (0.5, 0.9, 2.0):
    copies = create(MODEL, n_units=200, spectral_radius=radius, leak=0.3)
    one = Simulation(copies, dt=1).run(until=300).trajectory_['units.r']
    two = Simulation(copies, dt=1, initial_state={'units.r': start}).run(until=300).trajectory_['units.r']
    distance.semilogy(np.arange(301), np.linalg.norm(one - two, axis=1) + 1e-16, label=f'spectral radius {radius}')
distance.set_xlabel('step t')
distance.set_ylabel('distance between the copies')
distance.set_title('The echo state property')
distance.legend()
fig.tight_layout()
plt.show()
