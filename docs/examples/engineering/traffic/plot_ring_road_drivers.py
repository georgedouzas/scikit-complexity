"""
# Ring-road driver agents

Thirty drivers on a circular highway accelerate towards the speed limit, but every speed carries a small random error,
and a driver who would reach the car ahead stops dead. Those stops set off jams that never clear, so the traffic never
reaches the speed limit.
"""

# %%
# The system is built from its class with 30 cars on a ring of 1000, the speed limit 40, and speed errors up
# to 2%, and runs for 300 steps. The road is 1000 long and 1 wide, so the blueprint frames the whole road with a
# margin above and below it and draws the cars as large dots, which bunch together where a jam holds them.

import matplotlib.pyplot as plt
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.traffic import RingRoadDriversSystem

SYSTEM = 'ring_road_drivers'

road = rrb.Spatial2DView(
    origin='/drivers',
    name='ring road',
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 1000.0), y_range=(-150.0, 150.0)),
    overrides={'drivers': rr.Points2D.from_fields(radii=9.0).visualizer()},
)
blueprint = rrb.Blueprint(
    road,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = RingRoadDriversSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=2)
trajectory = simulate(system, until=300, dt=1, observers=[observer], random_state=0)

# %%
# Jams keep the traffic slow
# --------------------------
# On a sparse ring of 10 cars the gaps are wide enough to absorb the speed errors, so the mean speed should reach the
# speed limit 40. With 30 or 60 cars, drivers keep stopping behind the car ahead, so the mean speed should level off
# far below the limit, lower for more cars.

fig, ax = plt.subplots(figsize=(7, 4.5))
for n_cars in (10, 30, 60):
    run = simulate(RingRoadDriversSystem(n_cars=n_cars), until=300, dt=1, random_state=1)
    speeds = [run.agents('drivers', frame).values['speed'].mean() for frame in range(len(run.times))]
    ax.plot(run.times, speeds, label=f'{n_cars} cars')
ax.axhline(40.0, color='black', linestyle='--', linewidth=0.8, label='speed limit')
ax.set_xlabel('step')
ax.set_ylabel('mean speed')
ax.set_title('Speed errors set off jams that keep traffic slow')
ax.legend()
plt.show()
