"""
# Solid waste system

A growing city generates waste that is collected, recycled, composted, or landfilled. The landfill fills faster as the
population grows, and once it nears its capacity the residual waste is dumped in the open.
"""

# %%
# The city starts with a million people growing at 2% a year, each generating 0.2 tonnes of waste a year, and the
# landfill holds two million tonnes. The run lasts 40 years. The blueprint shows the landfill and the dumped waste.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import loops
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.processes import SolidWasteSystem

SYSTEM = 'solid_waste'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/city/landfill', name='landfill'),
        rrb.TimeSeriesView(origin='/city/dumped', name='dumped'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=5)
trajectory = simulate(SolidWasteSystem(), observers=[observer])

# %%
# Filling the landfill
# --------------------
# The population growth is the one reinforcing loop of the system, and the collection, the sales of recycled material
# and compost, and the capacity adjustment are balancing loops. The growth makes the landfilling rise until the
# acceptance table closes the landfill, and the residual waste then turns to dumping.

polarities = loops(SolidWasteSystem()).polarities
reinforcing, balancing = polarities.count('reinforcing'), polarities.count('balancing')
times = np.asarray(trajectory.times)
fig, (stocks, flows) = plt.subplots(1, 2, figsize=(10, 4))
stocks.plot(times, trajectory.values('city.landfill'), label='landfill')
stocks.plot(times, trajectory.values('city.dumped'), label='dumped')
stocks.axhline(2e6, color='grey', linestyle=':', label='landfill capacity')
stocks.set_xlabel('years')
stocks.set_ylabel('tonnes')
stocks.set_title('Waste stocks')
stocks.legend()
for path, label in [
    ('city.recycling', 'recycling'),
    ('city.composting', 'composting'),
    ('city.landfilling', 'landfilling'),
    ('city.dumping', 'dumping'),
]:
    flows.plot(times, trajectory.values(path), label=label)
flows.set_xlabel('years')
flows.set_ylabel('tonnes per year')
flows.set_title(f'Waste flows, {reinforcing} reinforcing and {balancing} balancing loops')
flows.legend()
fig.tight_layout()
plt.show()
