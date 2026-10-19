"""
# Skipper-Schabel-Wilcox log-kill

Weekly doses of chemotherapy cut the tumour by a fixed fraction, and the tumour regrows between doses, so its size
falls in a sawtooth.
"""

# %%
# The model is created by name from the catalog: a tumour of a million cells grows at the breast cancer rate of 0.223
# a day, and a one-day dose every week kills at the rate 2. The observer saves every second frame with the default
# blueprint, whose primary view is the time series of the number of cells.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'skipper_log_kill'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=42)

# %%
# Logarithmic steps
# -----------------
# Every week the logarithm of the tumour changes by the same amount, the growth `7 r` minus the kill `a dose width`,
# whatever the size of the tumour. The weekly changes should all equal `7 * 0.223 - 2 = -0.439`.

times, cells = simulation.trajectory_.times, simulation.trajectory_['tumour.n']
weeks = np.isin(np.round(times, 6), 7.0 * np.arange(7))
steps = np.diff(np.log(cells[weeks]))
fig, (curve, change) = plt.subplots(1, 2, figsize=(10, 4))
curve.semilogy(times, cells)
curve.set_xlabel('time (days)')
curve.set_ylabel('tumour cells')
curve.set_title('Each dose cuts the tumour by a fixed fraction')
change.bar(np.arange(1, len(steps) + 1), steps)
change.axhline(7 * 0.223 - 2.0, color='black', linestyle='--', label='7 r - a dose width')
change.set_xlabel('week')
change.set_ylabel('change of log(n) over the week')
change.set_title('Equal logarithmic steps')
change.legend()
fig.tight_layout()
plt.show()
