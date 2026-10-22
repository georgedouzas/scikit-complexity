"""
# Satellite orbit transfer

Pushing only along the radius, a bang-bang controller raises the energy of a satellite while keeping its angular
momentum: the orbit swells from a tight ellipse to a wide one with the same angular momentum, and the controller stops
once the target energy is reached.
"""

# %%
# The model is created by name from the catalog at the periapsis of the orbit with energy -0.7 and angular momentum
# 0.7, with the thrust 0.1 and the target energy -0.2, and is integrated with RK4 for 20 time units. The blueprint shows
# the distance from the planet and the effective energy in one time series view.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'satellite_transfer'

transfer = rrb.TimeSeriesView(
    origin='/',
    contents=['/satellite/s', '/energy'],
    name='distance and energy',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    transfer,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(
    system,
    dt=0.05,
    solver=RK4(dt=0.005),
    observers=[observer],
    observables=describe(MODEL).observables_of(system),
).run(until=20)

# %%
# From one Kepler orbit to another
# --------------------------------
# The energy should rise from -0.7 to the target -0.2 in about 9.81 time units and then stay constant, as the thrust
# switches off. In the plane, the satellite should leave the inner ellipse and settle on the outer one, both with the
# angular momentum 0.7.

trajectory = simulation.trajectory_
s, theta = trajectory['satellite.s'], trajectory['satellite.theta']
energy = trajectory['energy']
fig, (orbit, rise) = plt.subplots(1, 2, figsize=(10, 4.5))
orbit.plot(s * np.cos(theta), s * np.sin(theta), label='satellite')
orbit.plot(0.0, 0.0, 'o', color='black', label='planet')
orbit.set_aspect('equal')
orbit.set_xlabel('x')
orbit.set_ylabel('y')
orbit.set_title('Transfer to a wider orbit')
orbit.legend()
rise.plot(trajectory.times, energy, label='effective energy H')
rise.axhline(-0.2, color='black', linestyle='--', linewidth=0.8, label='target -0.2')
rise.axvline(9.81, color='gray', linestyle=':', label='expected transfer time 9.81')
rise.set_xlabel('time')
rise.set_ylabel('energy')
rise.set_title('The energy rises to the target and stays')
rise.legend()
fig.tight_layout()
plt.show()
