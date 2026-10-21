"""
# Light-sensitive robots

Robots slow down in the light and speed up in the dark, but they react to the light they measured five time units
ago, and they shuttle from wall to wall of their well across the light at its centre. They cross the light at the fast
speed they measured in the dark and linger outside it at the slow speed they measured in the light, so the delay
drives them out of the light.
"""

# %%
# The model is created by name from the catalog with 20 robots that start at the centre of a well of radius 2, the light
# of width 1, and the sensory delay 5, and runs for 100 time units. The robots' positions are vector states, so the
# blueprint shows their distances along `x` on the range of the well, without a legend.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'light_seeking_robots'

positions = rrb.TimeSeriesView(
    origin='/robots/x',
    name='x of the robots',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
    axis_y=rrb.ScalarAxis(range=(-2.0, 2.0)),
    plot_legend=rrb.PlotLegend(visible=False),
)
blueprint = rrb.Blueprint(
    positions,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
solver = EulerMaruyama(dt=0.01)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=0.5, solver=solver, observers=[observer], random_state=0).run(until=100)

# %%
# The delay decides
# -----------------
# Without delay a robot slows down as soon as it enters the light. With a positive delay it reacts late, so the light
# intensity at the robots `exp(-r^2 / r0^2)`, averaged after a transient, should fall from about 0.17 for delays up to
# 1 to about 0.13 for delays of 3 and more, and with a long delay the robots should come near the centre less often.

delays = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
intensities, distances = [], {}
for delay in delays:
    run = Simulation(create(MODEL, n_robots=40, delay=delay), dt=0.5, solver=solver, random_state=1).run(until=150)
    trajectory = run.trajectory_
    r = np.hypot(trajectory['robots.x'], trajectory['robots.y'])[trajectory.times > 50]
    intensities.append(np.exp(-(r**2)).mean())
    distances[delay] = r.ravel()
fig, (light, radii) = plt.subplots(1, 2, figsize=(10, 4))
light.plot(delays, intensities, 'o-')
light.set_xlabel('sensory delay')
light.set_ylabel('mean light intensity at the robots')
light.set_title('A longer delay keeps the robots out of the light')
for delay in (0.0, 8.0):
    radii.hist(distances[delay], bins=20, range=(0, 2), histtype='step', density=True, label=f'delay {delay:g}')
radii.set_yscale('log')
radii.set_xlabel('distance from the light')
radii.set_ylabel('probability density')
radii.set_title('Distances after t = 50')
radii.legend()
fig.tight_layout()
plt.show()
