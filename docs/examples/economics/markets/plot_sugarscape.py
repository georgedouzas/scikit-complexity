"""
# Sugarscape

Foragers of random vision and metabolism roam a landscape with two peaks of regrowing sugar. The short-sighted and
the hungry starve, the population falls to the number the landscape can feed, and the wealth of the survivors becomes
highly unequal.
"""

# %%
# Four hundred foragers start on random cells of a 50 by 50 grid with 5 to 25 units of sugar, a vision of 1 to 6
# cells, and a metabolism of 1 to 4 units a step. Every step each moves to the richest free cell it sees along the
# four lattice directions, harvests it, and burns its metabolism, and every cell grows back one unit of sugar up to its
# capacity of 0 to 4. The view shows the sugar on the cells, from empty (purple) to full (yellow), and the foragers in
# white, which gather on the two peaks.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create
from skcomplex.models.economics.markets._sugarscape import gini

MODEL = 'sugarscape'

whole = rrb.TimeRangeBoundary.infinite()
landscape = rrb.Spatial2DView(
    origin='/',
    contents=['/patches/**', '/foragers/**'],
    name='landscape',
    overrides={
        'patches': rr.Points2D.from_fields(radii=0.4).visualizer(),
        'foragers': rr.Points2D.from_fields(colors=[240, 240, 240], radii=0.45).visualizer(),
    },
)
census = rrb.TimeSeriesView(
    origin='/reporters',
    contents=['/reporters/census.population'],
    name='population',
    axis_x=rrb.TimeAxis(view_range=rrb.TimeRange(start=whole, end=whole)),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(landscape, census, row_shares=[2, 1]),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, width=50, height=50, n_foragers=400, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=100)

# %%
# Carrying capacity and inequality
# --------------------------------
# Downey (2018 ch. 9) finds that the population drops quickly and then levels off at the carrying capacity of the
# landscape, and that the wealth of the survivors is skewed. The Gini coefficient of wealth should rise from that of
# the uniform initial endowments, about 0.22, to well above 0.3, and the Lorenz curve of the final wealth should sag
# far below the line of equality.

trajectory = simulation.trajectory_
times = trajectory.times
initial = gini(trajectory.agents('foragers', 0).values['sugar'])
inequality = np.r_[initial, trajectory['census.gini'][1:]]
wealth = np.sort(trajectory.agents('foragers', len(times) - 1).values['sugar'])
lorenz = np.r_[0.0, np.cumsum(wealth) / wealth.sum()]
fig, (population, gini_axis, lorenz_axis) = plt.subplots(1, 3, figsize=(13, 4))
population.plot(times[1:], trajectory['census.population'][1:])
population.set_xlabel('step')
population.set_ylabel('foragers')
population.set_title('The population falls to a carrying capacity')
gini_axis.plot(times, inequality)
gini_axis.axhline(0.3, color='grey', linestyle=':')
gini_axis.set_xlabel('step')
gini_axis.set_ylabel('Gini coefficient of wealth')
gini_axis.set_title('Wealth becomes unequal')
lorenz_axis.plot(np.linspace(0, 1, len(lorenz)), lorenz, label=f'final wealth, Gini {inequality[-1]:.2f}')
lorenz_axis.plot([0, 1], [0, 1], color='black', linestyle='--', label='equality')
lorenz_axis.set_xlabel('share of foragers')
lorenz_axis.set_ylabel('share of wealth')
lorenz_axis.set_title('Lorenz curve')
lorenz_axis.legend()
fig.tight_layout()
plt.show()
