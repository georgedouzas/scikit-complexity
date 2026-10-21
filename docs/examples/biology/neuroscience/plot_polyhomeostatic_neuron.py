"""
# Polyhomeostatic neuron

A neuron driven by a periodic input adapts its gain and its threshold until its firing rate follows not only a target
mean but a target distribution, the exponential one of maximal entropy. Its mean rate drifts to the target while gain
and threshold settle.
"""

# %%
# The model is created by name from the catalog with the target mean rate `mu = 0.2`. The observer saves every second
# step with the default blueprint, whose primary view is the time series of the potential, the gain, and the
# threshold.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'polyhomeostatic_neuron'

system = create(MODEL, mu=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=400)

# %%
# The firing rate approaches its target
# -------------------------------------
# The rate is the sigmoid `y = 1 / (1 + exp(-a (x - b)))` of the potential `x`. Its running mean over a period of the
# input should approach the target `mu`, and the histogram of the late rates should be skewed towards low rates, like
# the exponential target.

trajectory = simulation.trajectory_
x, gain, threshold = trajectory['neuron.x'], trajectory['neuron.a'], trajectory['neuron.b']
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
