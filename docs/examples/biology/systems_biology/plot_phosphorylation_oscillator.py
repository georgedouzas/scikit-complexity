"""
# Three-node phosphorylation oscillator

Three proteins switch between their unphosphorylated and phosphorylated forms in turn, A, then B, then C, and the
loop keeps oscillating with a period of about 4.2.
"""

# %%
# The model is created by name from the catalog and run for 40 time units from A unphosphorylated and B and C
# phosphorylated. The Michaelis constants are a hundredth of the totals, so every phosphorylation cycle acts as a
# sharp switch. The observer saves every fourth frame with the default blueprint, whose primary view is the time
# series of the three amounts.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'phosphorylation_oscillator'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=40)

# %%
# A negative feedback loop of switches
# ------------------------------------
# C removes A, A restores B, and B restores C, so the loop is a negative feedback of three steps. Each protein should
# flip between nearly 0 and nearly 1 a fixed time after the protein before it, and the three amounts should trace one
# closed orbit, the limit cycle, once the start is forgotten.

trajectory = simulation.trajectory_
times = trajectory.times
fig, (series, orbit) = plt.subplots(1, 2, figsize=(10, 4))
for name in ('A', 'B', 'C'):
    series.plot(times, trajectory[f'loop.{name}'], label=name)
series.set_xlabel('time')
series.set_ylabel('unphosphorylated fraction')
series.set_title('The proteins switch in turn')
series.legend()
late = times >= 10.0
orbit.plot(trajectory['loop.A'][late], trajectory['loop.B'][late])
crossings = times[1:][(trajectory['loop.A'][1:] > 0.5) & (trajectory['loop.A'][:-1] <= 0.5)]
orbit.set_xlabel('A')
orbit.set_ylabel('B')
orbit.set_title(f'Limit cycle, period {np.mean(np.diff(crossings)):.2f}')
fig.tight_layout()
plt.show()
