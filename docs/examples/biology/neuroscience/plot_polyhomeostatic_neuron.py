"""
# Polyhomeostatic neuron

A neuron driven by a periodic input adapts its gain and its threshold until its firing rate follows not only a target
mean but a target distribution, the exponential one of maximal entropy. Its mean rate drifts to the target while gain
and threshold settle.
"""

# %%
# The system is built from its class with the target mean rate `mu = 0.2`. The observer saves every second step with
# the default blueprint, whose primary view is the time series of the potential, the gain, and the threshold.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import PolyhomeostaticNeuronSystem

SYSTEM = 'polyhomeostatic_neuron'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(PolyhomeostaticNeuronSystem(mu=0.2), observers=[observer])

# %%
# The firing rate approaches its target
# -------------------------------------
# The rate is the sigmoid `y = 1 / (1 + exp(-a (x - b)))` of the potential `x`. Its running mean over a period of the
# input should approach the target `mu`, and the histogram of the late rates should be skewed towards low rates, like
# the exponential target.

x, gain, threshold = (
    trajectory.get_values('neuron.x'),
    trajectory.get_values('neuron.a'),
    trajectory.get_values('neuron.b'),
)
rate = 1 / (1 + np.exp(-gain * (x - threshold)))
window = round(2 * np.pi)
running = np.convolve(rate, np.ones(window) / window, mode='same')
fig, (series, histogram) = plt.subplots(1, 2, figsize=(11, 4))
series.plot(trajectory.times, rate, linewidth=0.6, label='rate')
series.plot(trajectory.times, running, label='running mean')
series.axhline(0.2, color='black', linestyle='--', label='target mu')
series.set_xlabel('time')
series.set_ylabel('firing rate')
series.set_title(f'Mean rate of the last half {rate[len(rate) // 2 :].mean():.3f}')
series.legend()
histogram.hist(rate[len(rate) // 2 :], bins=20, density=True)
histogram.set_xlabel('firing rate')
histogram.set_ylabel('density')
histogram.set_title('Rates of the last half: mostly low')
fig.tight_layout()
plt.show()
