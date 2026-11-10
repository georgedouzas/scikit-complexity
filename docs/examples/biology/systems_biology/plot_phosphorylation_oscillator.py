"""
# Three-node phosphorylation oscillator

Three proteins switch between their unphosphorylated and phosphorylated forms in turn, A, then B, then C, and the
loop keeps oscillating with a period of about 4.2.
"""

# %%
# The example builds `PhosphorylationOscillatorSystem` and runs it for 40 time units from A unphosphorylated and B and C
# phosphorylated. The Michaelis constants are a hundredth of the totals, so every phosphorylation cycle acts as a sharp
# switch. The observer saves every fourth frame with the default blueprint, whose primary view is the time series of the
# three amounts.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import PhosphorylationOscillatorSystem

SYSTEM = 'phosphorylation_oscillator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(PhosphorylationOscillatorSystem(), observers=[observer])

# %%
# A negative feedback loop of switches
# ------------------------------------
# C removes A, A restores B, and B restores C, so the loop is a negative feedback of three steps. Each protein should
# flip between nearly 0 and nearly 1 a fixed time after the protein before it, and the three amounts should trace one
# closed orbit, the limit cycle, once the start is forgotten.

times = trajectory.times
fig, (series, orbit) = plt.subplots(1, 2, figsize=(10, 4))
for name in ('A', 'B', 'C'):
    series.plot(times, trajectory.get_values(f'loop.{name}'), label=name)
series.set_xlabel('time')
series.set_ylabel('unphosphorylated fraction')
series.set_title('The proteins switch in turn')
series.legend()
late = times >= 10.0
orbit.plot(trajectory.get_values('loop.A')[late], trajectory.get_values('loop.B')[late])
crossings = times[1:][(trajectory.get_values('loop.A')[1:] > 0.5) & (trajectory.get_values('loop.A')[:-1] <= 0.5)]
orbit.set_xlabel('A')
orbit.set_ylabel('B')
orbit.set_title(f'Limit cycle, period {np.mean(np.diff(crossings)):.2f}')
fig.tight_layout()
plt.show()
