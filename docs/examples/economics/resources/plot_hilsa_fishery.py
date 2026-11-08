"""
# Hilsa fishery

Hilsa spawn in the rivers, grow in the ocean for two years, and return to spawn. The fishery catches the juveniles
leaving the rivers and the adults about to spawn. Under the present catch the population settles where the spawning
grounds fill, a lower catch of juveniles raises the stock, and a heavier catch of spawners drives it to collapse.
"""

# %%
# The run starts from 375 million spawners and lasts ten years, in months. The blueprint shows the millions of adults
# about to spawn and of juveniles in the rivers.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.resources import HilsaFisherySystem

SYSTEM = 'hilsa_fishery'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/hilsa/spawners', name='adults about to spawn', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/hilsa/juveniles', name='juveniles', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(HilsaFisherySystem(), observers=[observer])

# %%
# Management scenarios
# --------------------
# Each spawner replaces itself about 1.26 times at the present catch, so the stock grows until the nests are full and
# then holds. Catching fewer juveniles lets more of the same eggs reach maturity and should raise the stock and the
# catch of adults. Catching 90 percent of the spawners pushes the replacement below one, and the stock should decay
# year after year.

fig, (stock, catch) = plt.subplots(1, 2, figsize=(10, 4))
scenarios = {
    'present catch': {},
    'juvenile catch 0.2': {'juvenile_harvest': 0.2},
    'adult catch 0.9': {'adult_harvest': 0.9},
}
for name, changes in scenarios.items():
    scenario = trajectory if not changes else simulate(HilsaFisherySystem(**changes))
    times = np.asarray(scenario.times) / 12
    stock.plot(times, scenario.values('hilsa.adult_weight'), label=name)
    catch.plot(times, scenario.values('hilsa.adult_catch') * 0.82e3, label=name)
stock.set_xlabel('years')
stock.set_ylabel('tonnes')
stock.set_title('Weight of adults about to spawn')
stock.set_yscale('log')
stock.legend()
catch.set_xlabel('years')
catch.set_ylabel('tonnes per month')
catch.set_title('Catch of adults')
catch.set_yscale('log')
catch.legend()
fig.tight_layout()
plt.show()
