"""
# Langevin car-following toward a density-dependent speed

Fifty cars drive around a ring road at the density 0.5. The speeds of all cars fluctuate in a narrow band around the
desired speed 0.5 of that density, so the cars keep flowing at about half the free speed.
"""

# %%
# The cars start evenly spaced at the desired speed `u(rho) = 1 - rho`, relax to it at the rate `gamma = 3`, and feel
# noise of intensity `D1 rho` with `D1 = 0.05`, where `rho` is the inverse headway. The blueprint shows the speeds of
# all cars on the range from standstill to the free speed, without a legend.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'langevin_traffic'

speeds = rrb.TimeSeriesView(
    origin='/cars/v',
    name='speeds',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
    axis_y=rrb.ScalarAxis(range=(0.0, 1.0)),
    plot_legend=rrb.PlotLegend(visible=False),
)
blueprint = rrb.Blueprint(
    speeds,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=1.0, solver=EulerMaruyama(dt=0.01), observers=[observer], random_state=0).run(
    until=200,
)

# %%
# The fundamental diagram and the speed fluctuations
# ---------------------------------------------------
# Every car relaxes to `u(rho)`, so the mean flow of a ring road should follow the fundamental diagram `rho u(rho)`,
# and the speeds of an Ornstein-Uhlenbeck process with the noise intensity `D1 rho` should have the variance
# `D1 rho / gamma` if the headways were fixed. Short runs at several densities should match the fundamental diagram,
# and the speed variance should follow the line at low densities and rise above it at high densities, where the
# fluctuating headways add to the noise.

gamma, D1, length = 3.0, 0.05, 100.0
densities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
flows, variances = [], []
for density in densities:
    cars = create(MODEL, n_cars=round(density * length), length=length, gamma=gamma, D1=D1)
    run = Simulation(cars, dt=1.0, solver=EulerMaruyama(dt=0.01), random_state=1).run(until=100)
    speeds = run.trajectory_['cars.v']
    flows.append(density * speeds[25:].mean())
    variances.append(speeds[25:].var())
grid = np.linspace(0.0, 1.0, 100)
fig, (diagram, noise) = plt.subplots(1, 2, figsize=(10, 4))
diagram.plot(grid, grid * (1 - grid), color='black', label='rho u(rho)')
diagram.plot(densities, flows, 'o', label='simulation')
diagram.set_xlabel('density rho')
diagram.set_ylabel('flow')
diagram.set_title('Fundamental diagram')
diagram.legend()
noise.plot(grid, D1 * grid / gamma, color='black', label='D1 rho / gamma')
noise.plot(densities, variances, 'o', label='simulation')
noise.set_xlabel('density rho')
noise.set_ylabel('speed variance')
noise.set_title('Speed fluctuations grow with the density')
noise.legend()
fig.tight_layout()
plt.show()
