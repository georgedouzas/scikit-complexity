"""
# Adaptive response costs

Episodes of demand push a critical parameter of an organism away from its norm every 10 time units. An adaptive
response builds capacity under the repeated demand: the first excursion overshoots the damage threshold at 1, and
as the capacity grows, the later ones stay well below it. The view shows the excursions, a train of spikes that
shrink, and the capacity that climbs with every episode and decays a little between them.
"""

# %%
# The model is created by name from the catalog with the adaptive strategy and episodes of height 1 and width 2 every
# 10 time units. The run lasts 200 time units and the observer saves every tenth step. The blueprint shows the
# excursion `e` and the capacity `c` over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'adaptive_response_costs'
COSTS = ('response', 'restore', 'repair', 'prepare')
STRATEGIES = ('stimulus-response', 'adaptive', 'anticipatory')

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/organism',
        contents=['/organism/e', '/organism/c'],
        name='excursion and capacity',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, strategy='adaptive')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=10)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=200)

# %%
# The costs of the three levels of adaptivity
# -------------------------------------------
# Mobus's objective is the sum of the response, restore, and repair costs, with the preparedness cost amortized over
# the episodes. Adaptive response should cut the total below stimulus-response by almost removing the repair cost, at
# the price of upkeep, and anticipation, which avoids half of every episode, should cut the response and restore costs
# as well.

totals = {}
for strategy in STRATEGIES:
    trajectory = Simulation(create(MODEL, strategy=strategy), dt=0.1).run(until=200).trajectory_
    totals[strategy] = [trajectory[f'organism.{kind}_cost'][-1] for kind in COSTS]
fig, (bars, peaks) = plt.subplots(1, 2, figsize=(10, 4))
bottom = np.zeros(len(STRATEGIES))
for index, kind in enumerate(COSTS):
    heights = np.array([totals[strategy][index] for strategy in STRATEGIES])
    bars.bar(STRATEGIES, heights, bottom=bottom, label=f'{kind} cost')
    bottom += heights
bars.set_ylabel('cost over 20 episodes')
bars.set_title('Total cost falls with each level of adaptivity')
bars.legend()
times = simulation.trajectory_.times
excursion = simulation.trajectory_['organism.e']
episodes = np.arange(1.0, 200.0, 10.0)
maxima = [excursion[(times >= start) & (times < start + 10)].max() for start in episodes]
peaks.plot(np.arange(1, len(maxima) + 1), maxima, 'o-')
peaks.axhline(1.0, color='black', linestyle=':', label='damage threshold')
peaks.set_xlabel('episode')
peaks.set_ylabel('largest excursion')
peaks.set_title('Adaptive response keeps later episodes below the threshold')
peaks.legend()
fig.tight_layout()
plt.show()
