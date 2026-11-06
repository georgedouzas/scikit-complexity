"""
# Bathtub

A bathtub, one stock of water with an inflow from the faucet and an outflow through the drain. The drain empties the
tub at five gallons per minute, and after five minutes the faucet opens at the same rate, so the level falls and then
holds.
"""

# %%
# The tub starts with 50 gallons, and the run lasts ten minutes. The blueprint shows the water in the tub.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'bathtub'

blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/tub/water', name='water'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create_system(SYSTEM), dt=0.5, observers=[observer]).prepare().run(10)

# %%
# The stock integrates its flows
# ------------------------------
# The water is a stock and the faucet and the drain are flows, so the level changes at the faucet minus the drain.
# The level falls at five gallons per minute until the faucet opens, and then the two flows balance.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (stock, flows) = plt.subplots(1, 2, figsize=(10, 4))
stock.plot(times, trajectory.values('tub.water'))
stock.set_xlabel('minutes')
stock.set_ylabel('gallons')
stock.set_title('The water in the tub')
flows.plot(times, trajectory.values('tub.faucet'), label='faucet')
flows.plot(times, trajectory.values('tub.drain'), '--', label='drain')
flows.set_xlabel('minutes')
flows.set_ylabel('gallons per minute')
flows.set_title('The flows')
flows.legend()
fig.tight_layout()
plt.show()
