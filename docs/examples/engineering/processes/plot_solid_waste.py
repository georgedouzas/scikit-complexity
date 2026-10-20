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
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'solid_waste'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/city/landfill', name='landfill'),
        rrb.TimeSeriesView(origin='/city/dumped', name='dumped'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=5)
simulation = Simulation(create(MODEL), dt=0.1, observers=[observer]).run(40)

# %%
# Filling the landfill
# --------------------
# The population growth is the one reinforcing loop of the model, and the collection, the sales of recycled material
# and compost, and the capacity adjustment are balancing loops. The growth makes the landfilling rise until the
# acceptance table closes the landfill, and the residual waste then turns to dumping.

polarities = [loop.polarity for loop in loops(create(MODEL)).loops]
reinforcing, balancing = polarities.count('reinforcing'), polarities.count('balancing')
trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (stocks, flows) = plt.subplots(1, 2, figsize=(10, 4))
stocks.plot(times, trajectory['city.landfill'], label='landfill')
stocks.plot(times, trajectory['city.dumped'], label='dumped')
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
    flows.plot(times, trajectory[path], label=label)
flows.set_xlabel('years')
flows.set_ylabel('tonnes per year')
flows.set_title(f'Waste flows, {reinforcing} reinforcing and {balancing} balancing loops')
flows.legend()
fig.tight_layout()
plt.show()
