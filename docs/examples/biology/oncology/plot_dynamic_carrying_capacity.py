"""
# Gompertz growth with a vascular carrying capacity

The tumour first fills its initial carrying capacity, and then grows together with the capacity that the vessels on
its surface keep raising.
"""

# %%
# The model is created by name from the catalog with the breast cancer fit of Padmanabhan et al., an initial volume of
# 1 mm³ and an initial capacity of 12.7 mm³, and run for 20 days. The observer saves every frame with the default
# blueprint, whose primary view is the time series of the volume and the capacity.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'dynamic_carrying_capacity'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# Tumour and capacity
# -------------------
# Once the tumour has filled its capacity, `n` stays close to `K`, so `dK/dt = d n^(2/3)` gives `K^(1/3)` rising
# linearly at the rate `d / 3`, and the volume grows as the cube of time.

times = simulation.trajectory_.times
volume, capacity = simulation.trajectory_['tumour.n'], simulation.trajectory_['vessels.K']
fig, (curves, cube) = plt.subplots(1, 2, figsize=(10, 4))
curves.plot(times, volume, label='tumour n')
curves.plot(times, capacity, label='capacity K')
curves.set_xlabel('time (days)')
curves.set_ylabel('volume (mm³)')
curves.set_title('The tumour follows its capacity')
curves.legend()
late = times > 5.0
slope = np.polyfit(times[late], np.cbrt(capacity[late]), 1)[0]
cube.plot(times, np.cbrt(capacity), label=f'simulated, slope {slope:.3f}')
cube.plot(
    times[late],
    np.cbrt(capacity[late][0]) + 0.829 / 3 * (times[late] - times[late][0]),
    '--',
    color='black',
    label='slope d / 3',
)
cube.set_xlabel('time (days)')
cube.set_ylabel('K^(1/3)')
cube.set_title('The cube root of the capacity rises linearly')
cube.legend()
fig.tight_layout()
plt.show()
