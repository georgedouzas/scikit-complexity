"""
# Ramped stochastic fold

A noisy state sits on the lower branch of a fold while its control rises slowly from -1 to 1. The branch creeps up
and flattens as the control nears the fold at `2 / (3 sqrt(3)) = 0.385`, where it vanishes, and the state then jumps
to the upper branch: a tipping point. The view shows the state rising slowly with growing fluctuations and then
jumping up near the middle of the run.
"""

# %%
# The model is created by name from the catalog with `a = -1`, `b = 1`, the control `phi = -1 + 0.004 t` of Fang et
# al., and noise of amplitude 0.02. The run lasts 500 time units with steps of 0.01, and the observer saves every
# second time unit. The blueprint shows the state over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'ramped_stochastic_fold'
FOLD = 2 / (3 * np.sqrt(3))

solver = describe(MODEL).defaults['solver']
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/system/x', name='state x', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(system, dt=0.5, solver=solver, observers=[observer], random_state=0).run(until=500)

# %%
# Critical slowing down and the lag of tipping
# --------------------------------------------
# Near the fold the recovery rate `-3 x^2 + 1` of the lower branch tends to zero, so the fluctuations around the
# branch, the difference from the noiseless run, should grow in variance as the control approaches 0.385. A ramp faster
# than the relaxation should carry the control past the fold before the state tips, by a lag that grows with the rate.

times = simulation.trajectory_.times
quiet = Simulation(create(MODEL, sigma=0.0), dt=0.5, solver=solver).run(until=500).trajectory_['system.x']
residual = simulation.trajectory_['system.x'] - quiet
windows = [(start, start + 40.0) for start in np.arange(0.0, 320.0, 20.0)]
variances = [residual[(times >= start) & (times < end)].var() for start, end in windows]
centres = [-1 + 0.004 * (start + end) / 2 for start, end in windows]
rates = np.array([0.002, 0.004, 0.01, 0.02, 0.05, 0.1])
tipped = []
for rate in rates:
    run = Simulation(create(MODEL, rate=rate), dt=0.1, solver=solver, random_state=0).run(until=2.0 / rate)
    tipped.append(-1 + rate * run.trajectory_.times[np.argmax(run.trajectory_['system.x'] > 0)])
fig, (slowing, lags) = plt.subplots(1, 2, figsize=(10, 4))
slowing.semilogy(centres, variances, 'o-')
slowing.axvline(FOLD, color='black', linestyle=':', label='static fold')
slowing.set_xlabel('control phi')
slowing.set_ylabel('variance of the fluctuations')
slowing.set_title('Variance rises toward the fold')
slowing.legend()
lags.semilogx(rates, tipped, 'o-', label='control when x crosses 0')
lags.axhline(FOLD, color='black', linestyle=':', label='static fold')
lags.set_xlabel('ramp rate')
lags.set_ylabel('control at tipping')
lags.set_title('Faster ramps tip later')
lags.legend()
fig.tight_layout()
plt.show()
