"""
# Drift to low performance

An actor corrects the gap between its desired and actual performance, but perceives the performance worse than it
is, and lets the goal slip toward the perceived state. The goal follows the pessimistic view down, the corrective
action shrinks, and the performance drifts down: Meadows' eroding goals. Absolute standards stop the drift, and an
upbeat bias turns it into a drift toward high performance.
"""

# %%
# The actor underrates the performance by 10%, and the run lasts 100 years. The default blueprint shows the
# performance, the perceived state, and the desired state.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.organizations import DriftToLowPerformanceSystem

SYSTEM = 'drift_to_low_performance'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(DriftToLowPerformanceSystem(), observers=[observer])

# %%
# The trap and the two ways out
# -----------------------------
# With eroding goals the performance should decay exponentially. With absolute standards it should hold at its
# initial level, and with an upbeat bias of 10% it should grow.

runs = {
    'eroding goals': trajectory,
    'absolute standards': simulate(DriftToLowPerformanceSystem(absolute=True)),
    'upbeat bias': simulate(DriftToLowPerformanceSystem(bias=-0.1)),
}
fig, (performance, states) = plt.subplots(1, 2, figsize=(10, 4))
for name, trajectory in runs.items():
    performance.semilogy(trajectory.times, trajectory.get_values('performance.performance'), label=name)
performance.set_xlabel('years')
performance.set_ylabel('performance')
performance.set_title('Drift down, hold, or drift up')
performance.legend()
trajectory = runs['eroding goals']
for state in ('performance', 'perceived', 'desired'):
    states.plot(trajectory.times, trajectory.get_values(f'performance.{state}'), label=state)
states.set_xlabel('years')
states.set_title('The goal follows the pessimistic view')
states.legend()
fig.tight_layout()
plt.show()
