"""
# Leaky integrate-and-fire neuron

A constant current charges the membrane of the leaky integrate-and-fire neuron along an exponential curve. Each time
the potential reaches the threshold the neuron spikes and resets, which gives a sawtooth of regular spikes.
"""

# %%
# The model is created by name from the catalog with an input that drives `R I = 20 mV`, above the 15 mV between rest
# and threshold, and run for 100 ms. The observer saves every fourth frame with the default blueprint, whose primary
# view is the time series of the potential.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'integrate_and_fire'

system = create(MODEL, I=2.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=100)

# %%
# The firing-rate curve
# ---------------------
# When `reset = rest` the neuron fires with period `tau ln(R I / (R I - threshold + rest))` above the rheobase
# current `(threshold - rest) / R = 1.5 nA` and not at all below it. Rerunning the neuron at several currents, the
# simulated rates should fall on that curve.

currents = np.linspace(1.0, 5.0, 17)
rates = []
for current in currents:
    events = Simulation(create(MODEL, I=current), dt=0.1).run(until=200).trajectory_.events
    rates.append(1000 * sum(event.label == 'spike' for event in events) / 200)
grid = np.linspace(1.51, 5.0, 200)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(currents, rates, 'o', label='simulated')
ax.plot(grid, 1000 / (10 * np.log(10 * grid / (10 * grid - 15))), '--', color='black', label='closed form')
ax.axvline(1.5, color='grey', linestyle=':', label='rheobase')
ax.set_xlabel('input current (nA)')
ax.set_ylabel('firing rate (Hz)')
ax.set_title('Firing rate against input current')
ax.legend()
plt.show()
