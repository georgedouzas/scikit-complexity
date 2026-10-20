"""
# Car-dealer inventory with delays

Meadows' car dealership keeps ten days of sales on its lot. When customer demand rises by 10%, the dealer sees the new
sales only after a perception delay, closes the gap to the new desired inventory over a response delay, and receives
the cars after a delivery delay. The delays make the inventory fall, overshoot, and oscillate before it settles.
"""

# %%
# The lot starts with 200 cars and a demand of 20 cars a day, which steps to 22 on day 25. The run lasts 100 days. The
# blueprint shows the cars on the lot.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'car_dealer'

blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/dealer/inventory', name='cars on the lot'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1.0, observers=[observer]).run(100)

# %%
# The flows and the delays
# ------------------------
# Sales drain the lot and deliveries fill it. Deliveries lag the orders by the delivery delay and overshoot the new
# sales, so the inventory oscillates. A longer response delay damps the oscillation, while a shorter perception delay
# hardly changes it, the lesson of Meadows' experiments.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (flows, delays) = plt.subplots(1, 2, figsize=(10, 4))
flows.plot(times, trajectory['dealer.sales'], label='sales')
flows.plot(times, trajectory['dealer.orders'], ':', label='orders')
flows.plot(times, trajectory['dealer.deliveries'], '--', label='deliveries')
flows.set_xlabel('days')
flows.set_ylabel('cars per day')
flows.set_title('The flows of the lot')
flows.legend()
for label, settings in {
    'perception 5, response 3': {},
    'perception 2, response 3': {'perception_delay': 2.0},
    'perception 5, response 6': {'response_delay': 6.0},
}.items():
    run = Simulation(create(MODEL, **settings), dt=1.0).run(100)
    delays.plot(times, run.trajectory_['dealer.inventory'], label=label)
delays.set_xlabel('days')
delays.set_ylabel('cars')
delays.set_title('The inventory under other delays')
delays.legend()
fig.tight_layout()
plt.show()
