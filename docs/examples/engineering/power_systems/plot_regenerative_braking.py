"""
# Regenerative braking

A Faraday disk on the wheel turns the motion of the car into a current that charges a supercapacitor and brakes the
car. The current is strongest at first and fades as the car slows and the capacitor charges, until the capacitor
starts to push current back.
"""

# %%
# The system is built from its class for a car of 1000 kg at 20 m/s and a supercapacitor of 680 F, and runs
# for one minute. The blueprint shows the speed and the capacitor voltage in one time series view, leaving out the
# current, which is a hundred times larger.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.power_systems import RegenerativeBrakingSystem

SYSTEM = 'regenerative_braking'

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
system = RegenerativeBrakingSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, observers=[observer])

# %%
# Where the kinetic energy goes
# -----------------------------
# The kinetic energy `m v^2 / 2` of the car should fall while the energy `C V_s^2 / 2` of the capacitor rises. Their
# sum falls too, by the energy lost in the resistor and to friction, and the share of the lost kinetic energy that the
# capacitor holds should reach its maximum when the current changes sign.

m, C = 1000.0, 680.0
kinetic = m * trajectory.values('car.v') ** 2 / 2
stored = C * trajectory.values('car.Vs') ** 2 / 2
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
    trajectory.times[np.argmax(trajectory.values('car.I') < 0)],
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
