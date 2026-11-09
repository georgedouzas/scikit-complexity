"""
# Izhikevich neuron

With Izhikevich's parameters for chattering neurons, a constant current makes the neuron fire bursts of fast spikes
separated by silences: the reset potential is high, so each spike is soon followed by another until the recovery
variable has grown enough to stop them.
"""

# %%
# The system is built from its class with the chattering parameters `c = -50` and `d = 2` and run for 100 ms. The
# observer saves every fourth frame with the default blueprint, whose primary view is the time series of the potential
# and the recovery.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import IzhikevichSystem

SYSTEM = 'izhikevich'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(IzhikevichSystem(c=-50.0, d=2.0, I=10.0), observers=[observer], until=100)

# %%
# Firing patterns
# ---------------
# The same equations give different firing patterns with different reset parameters. With the default regular
# spiking parameters the intervals between spikes should be long and lengthen as the neuron adapts, while the
# chattering neuron alternates short intervals within a burst and long ones between bursts.

patterns = {'regular spiking (c = -65, d = 8)': (-65.0, 8.0), 'chattering (c = -50, d = 2)': (-50.0, 2.0)}
fig, (traces, intervals) = plt.subplots(1, 2, figsize=(11, 4))
for offset, (label, (c, d)) in enumerate(patterns.items()):
    trajectory = simulate(IzhikevichSystem(c=c, d=d, I=10.0))
    spikes = np.array([event.time for event in trajectory.events if event.label == 'spike'])
    traces.plot(trajectory.times, trajectory.get_values('neuron.v') - 120 * offset, label=label)
    intervals.plot(np.diff(spikes), 'o-', label=label)
traces.set_xlabel('time (ms)')
traces.set_ylabel('potential (mV, offset)')
traces.set_title('Two firing patterns of one system')
traces.legend(loc='lower right')
intervals.set_yscale('log')
intervals.set_xlabel('spike number')
intervals.set_ylabel('interval to the next spike (ms)')
intervals.set_title('Bursts alternate short and long intervals')
intervals.legend(loc='lower right')
fig.tight_layout()
plt.show()
