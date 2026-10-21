"""
# Pólya urn

Each draw adds another ball of the drawn colour, so early luck is reinforced: the share of red balls
in each of six urns wanders at first and then freezes at its own random limit.
"""

# %%
# Six urns start with one red and one blue ball and make 200 draws. The model logs the numbers of red and blue
# balls; after the run the share of red in every urn is logged as a time series, and the blueprint puts it first, as
# the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy import stats

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'polya_urn'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/share', name='share of red balls', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, n_urns=6), dt=1.0, observers=[observer], random_state=0).run(until=200)
trajectory = simulation.trajectory_
share = trajectory['urns.red'] / (trajectory['urns.red'] + trajectory['urns.blue'])
for urn in range(6):
    observer.recording_.send_columns(
        f'share/urn_{urn}',
        indexes=[rr.TimeColumn('sim_time', duration=trajectory.times)],
        columns=rr.Scalars.columns(scalars=share[:, urn]),
    )
observer.recording_.flush()

# %%
# Beta limits
# -----------
# The share of red balls should converge in every urn, and over 2000 urns its limit should follow the beta distribution
# `Beta(red, blue)`: uniform for one ball of each colour, and `Beta(2, 1)`, of density `2x`, when the urn starts with
# two red balls and one blue.

fig, (shares, limits) = plt.subplots(1, 2, figsize=(11, 4))
shares.plot(trajectory.times, share)
shares.set_xlabel('draw')
shares.set_ylabel('share of red balls')
shares.set_title('Each urn freezes at its own share')
grid = np.linspace(0, 1, 100)
for red, color in ((1, 'C0'), (2, 'C1')):
    urns = Simulation(create(MODEL, n_urns=2000, red=red), dt=1.0, random_state=1).run(until=200).trajectory_
    share = urns['urns.red'][-1] / (urns['urns.red'][-1] + urns['urns.blue'][-1])
    limits.hist(share, bins=20, range=(0, 1), density=True, alpha=0.5, color=color, label=f'red = {red}, blue = 1')
    limits.plot(grid, stats.beta.pdf(grid, red, 1), color=color)
limits.set_xlabel('share of red after 200 draws')
limits.set_ylabel('density')
limits.set_title('Limits follow Beta(red, blue)')
limits.legend()
fig.tight_layout()
plt.show()
