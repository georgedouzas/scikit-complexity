"""
# Cardiovascular system

The left ventricle stiffens and relaxes with every beat, pumping blood into the arteries, through the body, and back
through the veins. The ventricular volume falls in each systole and refills in each diastole, while the arterial
volume pulses with the heartbeat and the total volume of blood stays constant.
"""

# %%
# The example builds `CardiovascularSystem` at 75 beats per minute and runs it for about six beats. The observer saves
# every frame with the default blueprint, whose primary view is the time series of the three volumes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import CardiovascularSystem

SYSTEM = 'cardiovascular'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(CardiovascularSystem(heart_rate=75.0), observers=[observer], until=5)

# %%
# Stroke volume and conservation
# ------------------------------
# The ventricle ejects a stroke volume of blood into the arteries at every beat, so the arterial volume should rise as
# the ventricular volume falls. Blood only moves between compartments, so the total should be conserved to the error
# of the solver.

ventricle, arteries, veins = (
    trajectory.get_values('ventricle.V'),
    trajectory.get_values('arteries.V'),
    trajectory.get_values('veins.V'),
)
total = ventricle + arteries + veins
fig, (volumes, conservation) = plt.subplots(1, 2, figsize=(10, 4))
volumes.plot(trajectory.times, ventricle, label='ventricle')
volumes.plot(trajectory.times, arteries - arteries.mean() + ventricle.mean(), label='arteries, shifted')
volumes.set_xlabel('time (s)')
volumes.set_ylabel('volume (ml)')
volumes.set_title(f'Stroke volume {np.ptp(ventricle[trajectory.times > 2]):.0f} ml')
volumes.legend()
conservation.plot(trajectory.times, total - total[0])
conservation.set_xlabel('time (s)')
conservation.set_ylabel('change of the total volume (ml)')
conservation.set_title('The blood volume is conserved')
fig.tight_layout()
plt.show()
