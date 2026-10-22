"""
# Tragedy of the commons

Twenty herders graze their cows on one meadow, and each adds a cow whenever a cow pays more than its upkeep. The herd
outgrows the grass, the meadow is grazed bare within a dozen steps, the cows starve, and the herd crashes after it,
in cycles of boom and bust.
"""

# %%
# The meadow is a 20 by 20 grid of cells, each holding up to 2 units of grass that grow back logistically at the rate
# 0.2. Every step a matching law pairs every cow with the free cell of most grass in its reach, the cow eats up to one
# unit there, burns half a unit, and earns its herder milk for what it ate, and a cow that runs out of energy dies.
# Then every herder whose cows earned more than their upkeep adds one. Nobody regulates the herd. The blueprint shows
# the grass on the meadow and the size of the herd.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'tragedy_of_the_commons'
STEPS = 100

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/grass', name='grass on the meadow', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(origin='/herd', name='cows', axis_x=rrb.TimeAxis(view_range=whole_run)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=STEPS)

# %%
# A regulator saves the meadow
# ----------------------------
# Hardin (1968) argued that a commons open to all is overused, since every user gains the whole benefit of one more
# animal but bears only a share of its cost (Hamill and Gilbert 2016 sec. 10.2). Without regulation the grass should
# fall below a tenth of its capacity and the herd should crash after it. A regulator that halves a quota of cows per
# herder whenever the grass falls below half of its capacity, and raises it by one cow above 80%, should keep the
# grass above a quarter of its capacity and a steady herd.

regulated_system = create(MODEL, quota=2.0, random_state=0)
regulated = Simulation(
    regulated_system,
    random_state=0,
    observables=describe(MODEL).observables_of(regulated_system),
).run(until=STEPS)
fig, (grass, herd) = plt.subplots(1, 2, figsize=(11, 4))
for label, run in (('open access', simulation), ('regulated quota', regulated)):
    trajectory = run.trajectory_
    grass.plot(trajectory.times, trajectory['grass'] / trajectory['grass'][0], label=label)
    herd.plot(trajectory.times, trajectory['herd'], label=label)
grass.axhline(0.1, color='grey', linestyle=':', label='a tenth of the capacity')
grass.set_xlabel('step')
grass.set_ylabel('grass as a share of the capacity')
grass.set_title('Open access grazes the meadow bare')
grass.legend()
herd.set_xlabel('step')
herd.set_ylabel('cows')
herd.set_title('The herd booms and crashes')
herd.legend()
fig.tight_layout()
plt.show()
