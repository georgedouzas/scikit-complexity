"""
# Oregonator

The Belousov-Zhabotinsky reaction oscillates by relaxation: bromide `y` builds up in a sharp spike, decays slowly
while bromous acid and the catalyst stay low, and spikes again once it falls below a threshold, at a regular period.
"""

# %%
# The example builds `OregonatorSystem` with the standard parameters and integrates it with its tuned LSODA solver,
# since the small `epsilon` and `delta` make it stiff. The run covers five periods, and the default blueprint shows the
# three concentrations in one time series view, where the spikes of bromide dominate.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import events
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.kinetics import OregonatorSystem

SYSTEM = 'oregonator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=3)
trajectory = simulate(OregonatorSystem(), observers=[observer])

# %%
# Limit cycle and period
# ----------------------
# On the relaxation limit cycle the concentrations span orders of magnitude, so the orbit is drawn on logarithmic
# axes. After the first transient spike, every spike of bromide above ten should start one period after the previous
# one, so the intervals between spikes should be equal.

times, x, y, z = (
    trajectory.times,
    trajectory.get_values('reactor.x'),
    trajectory.get_values('reactor.y'),
    trajectory.get_values('reactor.z'),
)
spikes = events(trajectory, variables=['reactor.y'], threshold=10.0).starts
periods = np.diff(spikes[1:])
fig, (orbit, intervals) = plt.subplots(1, 2, figsize=(10, 4))
cycle = times > spikes[1]
orbit.loglog(x[cycle], y[cycle])
orbit.set_xlabel('bromous acid x')
orbit.set_ylabel('bromide y')
orbit.set_title('Relaxation limit cycle')
intervals.plot(np.arange(1, len(periods) + 1), periods, 'o-')
intervals.axhline(periods.mean(), color='black', linestyle='--', label=f'mean period {periods.mean():.2f}')
intervals.set_ylim(0, 1.5 * periods.max())
intervals.set_xlabel('spike')
intervals.set_ylabel('time to the next spike')
intervals.set_title('The spikes repeat at one period')
intervals.legend()
fig.tight_layout()
plt.show()
