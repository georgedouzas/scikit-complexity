"""
# Car wash

Chesney's hand car wash: passing cars turn in at random, queue for a wash, and some stay for a valet, while the
workers serve the front car of each area. The till keeps a shadow account of the revenue the workers' labour earns.
Demand, not labour, sets the revenue, so extra workers only shorten the queue.
"""

# %%
# Every step is one second, and the wash is open for an eight-hour day with four workers. A car passes every three
# seconds on average and one in two hundred turns in, a wash takes 525 person-seconds and a valet 1750. The blueprint
# shows the number of cars on the forecourt, sampled every 100 seconds.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'car_wash'
DAY, P_PASS, P_ENTER, P_VALET, WASH, VALET = 8 * 3600, 0.33, 0.005, 0.5, 4.0, 10.0

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters',
        contents=['/reporters/cars.count'],
        name='cars on the forecourt',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=100)
simulation = Simulation(create(MODEL, n_workers=4), observers=[observer], random_state=0).run(until=DAY)

# %%
# Revenue against demand
# ----------------------
# Cars turn in at the rate `p_pass p_enter`, about six an hour, and each brings `(1 - p_valet) 4 + p_valet 10 = 7`
# on average, so the till should grow along the line of 41.6 an hour. One or two workers cannot keep up, the queue
# grows and drivers drive past, but from three workers on the daily revenue should saturate at the demand, while the
# forecourt keeps emptying and the revenue per worker keeps falling.

trajectory = simulation.trajectory_
frames = np.arange(0, len(trajectory.times), 100)
revenue = [trajectory.agents('till', frame).values['revenue'][0] for frame in frames]
expected = P_PASS * P_ENTER * ((1 - P_VALET) * WASH + P_VALET * VALET)
workers = np.array([1, 2, 3, 4, 6])
days = [Simulation(create(MODEL, n_workers=n), random_state=0).run(until=DAY).trajectory_ for n in workers]
earned = np.array([day.agents('till', len(day.times) - 1).values['revenue'][0] for day in days])
fig, (till, labour) = plt.subplots(1, 2, figsize=(10, 4))
till.plot(trajectory.times[frames] / 3600, revenue, label='till')
till.plot(trajectory.times[frames] / 3600, expected * trajectory.times[frames], 'k--', lw=0.8, label='expected')
till.set_xlabel('hour')
till.set_ylabel('cumulative revenue')
till.set_title('Revenue follows the arrivals')
till.legend()
labour.plot(workers, earned / workers, 'o-', label='revenue per worker')
labour.axhline(expected * DAY, color='grey', linestyle=':', label='expected daily revenue')
labour.plot(workers, earned, 's-', label='daily revenue')
cars = labour.twinx()
cars.plot(workers, [day['cars.count'].mean() for day in days], '^--', color='C3')
cars.set_ylabel('mean cars on the forecourt', color='C3')
labour.set_xlabel('number of workers')
labour.set_ylabel('revenue')
labour.set_title('Revenue saturates at the demand')
labour.legend(fontsize=8)
fig.tight_layout()
plt.show()
