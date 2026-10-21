"""
# Gyroscope in a gimbal

A torque on the axis of the inner ring barely tilts the fast rotor. Instead the gimbal turns about the fixed axis at a
steady slow rate, the precession, while the inner ring nods rapidly by a tiny angle, the nutation.
"""

# %%
# The model is created by name from the catalog with the rotor spinning at 50 in an inner ring tilted by 0.3 and a
# unit torque on the axis of the inner ring, and runs for 10 time units. The blueprint shows the angle of the outer
# ring, which falls at the steady precession rate, over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'gimballed_gyroscope'

precession = rrb.TimeSeriesView(
    origin='/gyroscope/alpha',
    name='angle of the outer ring',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    precession,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=10)

# %%
# Precession and nutation
# -----------------------
# Balancing the torque with the gyroscopic torque gives the mean precession rate `-torque / (C spin cos beta)`, which
# shrinks as the rotor spins faster. The rate itself nutates fast between zero and twice its mean, so it must be
# averaged over the run, not read at a few instants.

trajectory = simulation.trajectory_
early = trajectory.times <= 0.5
spins = np.array([25.0, 50.0, 100.0, 200.0])
rates = [
    Simulation(create(MODEL, spin=spin), dt=0.01).run(until=10).trajectory_['gyroscope.alpha'][-1] / 10
    for spin in spins
]
grid = np.linspace(20.0, 220.0, 100)
fig, (nutation, law) = plt.subplots(1, 2, figsize=(10, 4))
nutation.plot(trajectory.times[early], trajectory['gyroscope.alpha_d1'][early], label="precession rate alpha'")
nutation.axhline(-1 / (2 * 50 * np.cos(0.3)), color='black', linestyle='--', label='mean rate')
nutation.set_xlabel('time')
nutation.set_ylabel("alpha'")
nutation.set_title('Nutation around the mean precession')
nutation.legend()
law.loglog(spins, -np.array(rates), 'o', label='simulation')
law.loglog(grid, 1 / (2 * grid * np.cos(0.3)), color='black', label='torque / (C spin cos beta)')
law.set_xlabel('spin of the rotor')
law.set_ylabel('mean precession rate')
law.set_title('Faster rotors precess more slowly')
law.legend()
fig.tight_layout()
plt.show()
