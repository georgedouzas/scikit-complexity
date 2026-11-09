"""
# Gas-kinetic-based traffic

Drivers of the gas-kinetic-based traffic system brake on the traffic at an interaction point ahead of them, read at a
distance that grows with their own speed. Between about 24 and 51 vehicles per kilometre, uniform traffic is unstable:
a small dipole of density grows into stop-and-go waves, jams of about 57 vehicles per kilometre whose fronts travel
against the traffic.
"""

# %%
# The system is built from its class on a ring of 10 kilometres in 200 cells, at 35 vehicles per kilometre
# with a perturbation of 2, with the calibrated parameters of Treiber, Hennecke, and Helbing. Lengths are in kilometres
# and times in hours, and SSPRK integrates it for one hour. The default blueprint puts the history of the density first,
# where the jams are the bright bands that lean against the traffic.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import SSPRKSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.engineering.traffic import GktTrafficSystem

SYSTEM = 'gkt_traffic'

system = GktTrafficSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, dt=1 / 60, solver=SSPRKSolver(cfl=0.4), observers=[observer])

# %%
# Fronts that move upstream
# -------------------------
# The profile should travel without changing shape, so the shift that best aligns two profiles 10 minutes apart gives
# its velocity. The review of Helbing reports upstream fronts of 15 plus or minus 5 kilometres per hour on real
# freeways, and this system should give about 20, from the jump between the jam and its outflow. The density should stay
# conserved.

rho = trajectory.get_lattice('road.rho')
minutes = trajectory.times * 60
shifts = []
for earlier, later in zip(rho[30:-10], rho[40:], strict=True):
    correlation = np.fft.irfft(np.conj(np.fft.rfft(earlier - earlier.mean())) * np.fft.rfft(later - later.mean()))
    shift = int(np.argmax(correlation))
    shifts.append((shift - 200 if shift > 100 else shift) * 0.05)
velocity = np.mean(shifts) * 6

fig, (history, profile) = plt.subplots(1, 2, figsize=(10, 4))
image = history.imshow(rho, aspect='auto', origin='lower', extent=(0, 10, 0, 60))
fig.colorbar(image, ax=history, label='vehicles per kilometre')
history.set_xlabel('position on the ring (km)')
history.set_ylabel('time (min)')
history.set_title(f'Fronts move at {velocity:.1f} km/h')
profile.plot((np.arange(200) + 0.5) * 0.05, rho[-1])
profile.axhline(rho[0].mean(), color='black', linestyle='--', linewidth=0.8, label='mean density')
profile.set_xlabel('position on the ring (km)')
profile.set_ylabel('density')
profile.set_title('Jams between stretches of free traffic')
profile.legend()
fig.tight_layout()
plt.show()
