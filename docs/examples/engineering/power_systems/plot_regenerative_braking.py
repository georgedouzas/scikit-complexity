"""
# Regenerative braking

A Faraday disk on the wheel turns the motion of the car into a current that charges a supercapacitor and brakes the
car. The current is strongest at first and fades as the car slows and the capacitor charges, until the capacitor
starts to push current back.
"""

# %%
# The model is created by name from the catalog for a car of 1000 kg at 20 m/s and a supercapacitor of 680 F, and runs
# for one minute. The blueprint shows the speed and the capacitor voltage in one time series view, leaving out the
# current, which is a hundred times larger.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'regenerative_braking'

braking = rrb.TimeSeriesView(
    origin='/',
    contents=['/car/v', '/car/Vs'],
    name='speed and capacitor voltage',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    braking,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=60)

# %%
# Where the kinetic energy goes
# -----------------------------
# The kinetic energy `m v^2 / 2` of the car should fall while the energy `C V_s^2 / 2` of the capacitor rises. Their
# sum falls too, by the energy lost in the resistor and to friction, and the share of the lost kinetic energy that the
# capacitor holds should reach its maximum when the current changes sign.

m, C = 1000.0, 680.0
trajectory = simulation.trajectory_
kinetic = m * trajectory['car.v'] ** 2 / 2
stored = C * trajectory['car.Vs'] ** 2 / 2
recovered = np.divide(stored, kinetic[0] - kinetic, out=np.zeros_like(stored), where=kinetic[0] > kinetic)
fig, (energies, share) = plt.subplots(1, 2, figsize=(10, 4))
energies.plot(trajectory.times, kinetic / 1e3, label='kinetic energy of the car')
energies.plot(trajectory.times, stored / 1e3, label='energy in the capacitor')
energies.set_xlabel('time (s)')
energies.set_ylabel('energy (kJ)')
energies.set_title('Kinetic energy moves into the capacitor')
energies.legend()
share.plot(trajectory.times, recovered)
share.axvline(
    trajectory.times[np.argmax(trajectory['car.I'] < 0)],
    color='black',
    linestyle='--',
    label='current reverses',
)
share.set_xlabel('time (s)')
share.set_ylabel('stored / kinetic energy lost')
share.set_title('Share of the braking energy recovered')
share.legend()
fig.tight_layout()
plt.show()
