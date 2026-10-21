"""
# Huygens clocks

Two pendulum clocks hang from supports that can move and are joined by a spring. One starts swinging and the other at
rest; the motion of the supports passes energy between them until both swing with the same amplitude in antiphase,
the sympathy of clocks that Huygens observed in 1665.
"""

# %%
# The model is created by name from the catalog with two identical clocks, the first released from 0.2, and runs for
# 100 time units. The blueprint shows the angles of both pendulums over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'huygens_clocks'

angles = rrb.TimeSeriesView(
    origin='/',
    contents=['/clock_0/theta', '/clock_1/theta'],
    name='pendulum angles',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    angles,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=8)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=100)

# %%
# Antiphase synchronization
# -------------------------
# The correlation of the two angles over a sliding window of ten time units should move from positive values, while
# the clocks swing independently, to -1, the antiphase state, where the supports stay nearly still and both amplitudes
# settle at the limit cycle of the escapement.

trajectory = simulation.trajectory_
first, second = trajectory['clock_0.theta'], trajectory['clock_1.theta']
window = 200
starts = np.arange(0, len(first) - window, 20)
correlations = [np.corrcoef(first[i : i + window], second[i : i + window])[0, 1] for i in starts]
fig, (sync, amplitude) = plt.subplots(1, 2, figsize=(10, 4))
sync.plot(trajectory.times[starts + window // 2], correlations)
sync.axhline(-1.0, color='black', linestyle='--', linewidth=0.8, label='antiphase')
sync.set_ylim(-1.1, 1.1)
sync.set_xlabel('time')
sync.set_ylabel('correlation of the two angles')
sync.set_title('The clocks lock in antiphase')
sync.legend()
for name, label in [('clock_0.theta', 'first clock'), ('clock_1.theta', 'second clock')]:
    values = np.abs(trajectory[name])
    envelope = [values[i : i + window].max() for i in starts]
    amplitude.plot(trajectory.times[starts + window // 2], envelope, label=label)
amplitude.set_xlabel('time')
amplitude.set_ylabel('amplitude of the angle')
amplitude.set_title('Energy passes between the clocks until they match')
amplitude.legend()
fig.tight_layout()
plt.show()
