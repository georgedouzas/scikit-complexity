"""
# Delayed LIF network

Ten leaky integrate-and-fire neurons, each driven above threshold, fire on their own at a regular rate. Their spikes
reach the neurons they connect to after a delay and kick their potential up, so each neuron fires a little faster
than it would alone.
"""

# %%
# The model is created by name from the catalog with excitatory links of 0.5 mV, a delay of 1.5 ms, and a drive of
# 25 mV. The observer saves every frame with the default blueprint, whose primary view is the time series of the
# potentials.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'delayed_lif_network'

system = create(MODEL, n_neurons=10, J=0.5, D=1.5, mu=25.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# Excitation speeds the firing up
# -------------------------------
# A spike event resets the potential of the neuron that fired, which identifies it. Alone, every neuron should fire
# with the period `tau ln((mu - reset) / (mu - threshold))`, and coupled by excitatory links its rate should be at
# least as high, more so for neurons with more inputs.


def spikes_of(trajectory):
    """Return the time and the neuron of every spike."""
    times, neurons = [], []
    for event in trajectory.events:
        if event.label == 'spike':
            potentials_before, potentials_after = event.before[::2], event.after[::2]
            for neuron in np.nonzero(potentials_after < potentials_before - 1)[0]:
                times.append(event.time)
                neurons.append(neuron)
    return np.array(times), np.array(neurons)


coupled = spikes_of(simulation.trajectory_)
alone = spikes_of(Simulation(create(MODEL, J=0.0, random_state=0), dt=0.5).run(until=100).trajectory_)
fig, (raster, rates) = plt.subplots(1, 2, figsize=(11, 4))
raster.plot(*coupled, '|', color='black', markersize=10)
raster.set_xlabel('time (ms)')
raster.set_ylabel('neuron')
raster.set_title('Spikes of the coupled network')
counts_alone = np.bincount(alone[1], minlength=10)
counts_coupled = np.bincount(coupled[1], minlength=10)
rates.bar(np.arange(10) - 0.2, counts_alone * 10, width=0.4, label='J = 0')
rates.bar(np.arange(10) + 0.2, counts_coupled * 10, width=0.4, label='J = 0.5 mV')
rates.set_xlabel('neuron')
rates.set_ylabel('firing rate (Hz)')
rates.set_title(f'Mean rate {counts_alone.mean() * 10:.0f} Hz alone, {counts_coupled.mean() * 10:.0f} Hz coupled')
rates.legend()
fig.tight_layout()
plt.show()
