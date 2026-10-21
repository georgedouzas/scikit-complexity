"""
# Impulse-driven RLC circuit

Every two time units a current impulse kicks the circuit, which then rings down until the next kick. After a few
kicks the response repeats exactly from one impulse to the next: a periodic, non-sinusoidal waveform.
"""

# %%
# The model is created by name from the catalog with `L = C = 1`, `R = 0.1`, and impulses of strength 1 every
# `T = 2`, and runs for 150 time units, long enough for the weak damping to remove the transient. The blueprint
# shows only the response `x`, leaving out its rate and the clock of the impulses.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'impulsive_circuits'

response = rrb.TimeSeriesView(
    origin='/',
    contents=['/circuit/x'],
    name='response x',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    response,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=6)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=150)

# %%
# The periodic response
# ---------------------
# Just before every impulse the state should converge to the fixed point of the map over one period, so the values of
# `x` and `x'` sampled once per period should settle, and the last periods of the response should lie on each other.

trajectory = simulation.trajectory_
x, rate = trajectory['circuit.x'], trajectory['circuit.x_d1']
per_period = round(2.0 / 0.1)
starts = np.arange(1, len(x) // per_period) * per_period - 1
fig, (strobe, shape) = plt.subplots(1, 2, figsize=(10, 4))
strobe.plot(np.arange(len(starts)), x[starts], 'o-', label='x')
strobe.plot(np.arange(len(starts)), rate[starts], 's-', label="x'")
strobe.set_xlabel('period')
strobe.set_ylabel('value just before the impulse')
strobe.set_title('The stroboscopic state converges')
strobe.legend()
phase = np.arange(per_period) * 0.1
for index in range(len(starts) - 5, len(starts)):
    shape.plot(phase, x[starts[index] + 1 : starts[index] + 1 + per_period], linewidth=1)
shape.set_xlabel('time since the impulse')
shape.set_ylabel('x')
shape.set_title('The last five periods coincide')
fig.tight_layout()
plt.show()
