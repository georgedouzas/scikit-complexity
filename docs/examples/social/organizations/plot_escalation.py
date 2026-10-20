"""
# Escalation

Two parties each want to hold 10% more than the other. Each one sees the other's stock after a perception delay and
builds up to its target over an adjustment time. The loop through both stocks is reinforcing, so both stocks grow
exponentially, an arms race.
"""

# %%
# The parties start with 10 and 10.5 units, and the run lasts 100 years. The blueprint shows the two stocks.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'escalation'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/race/a', name='first party a'),
        rrb.TimeSeriesView(origin='/race/b', name='second party b'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=0.5, observers=[observer]).run(100)

# %%
# The growth rate of the race
# ---------------------------
# Once both parties build, the stocks grow as `exp(r t)`, where `r` solves `r T = (1 + m) / (1 + r tau) - 1` for the
# margin `m = 0.1`, the adjustment time `T = 2`, and the perception delay `tau = 1`. On a log scale the stocks become
# parallel lines of slope `r`.

margin, adjustment, perception = 0.1, 2.0, 1.0
quadratic = (adjustment * perception, adjustment + perception, -margin)
rate = max(np.roots(quadratic).real)
trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (stocks, gap) = plt.subplots(1, 2, figsize=(10, 4))
stocks.semilogy(times, trajectory['race.a'], label='first party')
stocks.semilogy(times, trajectory['race.b'], label='second party')
stocks.semilogy(
    times,
    trajectory['race.a'][-1] * np.exp(rate * (times - times[-1])),
    ':',
    color='black',
    label='exp(r t)',
)
stocks.set_xlabel('years')
stocks.set_ylabel('stock')
stocks.set_title(f'Exponential growth at r = {rate:.4f}')
stocks.legend()
gap.plot(times, trajectory['race.b'] / trajectory['race.a'])
gap.set_xlabel('years')
gap.set_ylabel('b / a')
gap.set_title('The ratio of the stocks')
fig.tight_layout()
plt.show()
