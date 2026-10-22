"""
# Couzin leadership

A shoal of fish swims together by avoiding close neighbours and following farther ones. Five of the fifty fish also
know a preferred direction, to the right, and nobody knows who they are. They are enough to turn the shoal: the view
shows the mean horizontal component of the headings, which climbs from about zero for random headings to about 0.8,
as the shoal turns to swim to the right.
"""

# %%
# The model is created by name from the catalog with 50 fish, of which 10% are informed with the weight
# `omega = 0.5` of the preferred direction at angle 0, on a torus of side 50. The primary view shows the observable
# `heading_x`, the mean of the horizontal headings, over the whole run. The view of the torus beside it draws the fish
# larger than the model's visual does, with the informed fish in a lighter colour.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.system import Observable
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'couzin_leadership'
SIZE = 50.0


def mean_heading(trajectory, frame):
    """Return the angle of the mean heading of the shoal at a frame."""
    fish = trajectory.agents('shoal', frame)
    return np.arctan2(np.mean(fish.values['vy']), np.mean(fish.values['vx']))


shoal = rrb.Spatial2DView(
    origin='/shoal',
    name='shoal',
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, SIZE), y_range=(-SIZE, 0.0)),
    overrides={'shoal': rr.Points2D.from_fields(radii=0.6).visualizer()},
)
# The view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(rrb.TimeSeriesView(origin='/heading_x', name='mean horizontal heading', axis_x=whole_run), shoal),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, n_fish=50, informed=0.1, size=SIZE, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
heading_x = Observable('heading_x', group='shoal', variable='vx')
simulation = Simulation(system, dt=1, observers=[observer], observables=[heading_x], random_state=0).run(until=150)

# %%
# Accuracy and the size of the informed minority
# ----------------------------------------------
# Couzin et al. (2005) found that the accuracy of a group, how closely its heading matches the preferred direction,
# grows with the fraction of informed individuals. The cosine of the shoal's heading
# in the run above should approach 1, and the mean cosine over shoals of 20 fish, without informed fish near 0, should
# rise with the informed fraction.

headings = [mean_heading(simulation.trajectory_, frame) for frame in range(len(simulation.trajectory_.times))]
fractions = [0.0, 0.1, 0.2, 0.4]
accuracy = []
for fraction in fractions:
    cosines = []
    for seed in range(3):
        run = Simulation(create(MODEL, n_fish=20, informed=fraction, random_state=seed), dt=1, random_state=seed)
        trajectory = run.run(until=120).trajectory_
        cosines.extend(np.cos(mean_heading(trajectory, frame)) for frame in range(80, 121))
    accuracy.append(np.mean(cosines))

fig, (turning, accuracies) = plt.subplots(1, 2, figsize=(10, 4))
turning.plot(simulation.trajectory_.times, np.cos(headings))
turning.set_ylim(-1.05, 1.05)
turning.set_xlabel('step')
turning.set_ylabel('cosine of heading to the preferred direction')
turning.set_title('Five informed fish turn fifty')
accuracies.plot(fractions, accuracy, 'o-')
accuracies.set_ylim(-1.05, 1.05)
accuracies.set_xlabel('fraction of informed fish')
accuracies.set_ylabel('accuracy, mean cosine of heading')
accuracies.set_title('Accuracy grows with the informed minority')
fig.tight_layout()
plt.show()
