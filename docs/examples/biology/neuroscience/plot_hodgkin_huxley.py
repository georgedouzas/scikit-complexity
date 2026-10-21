"""
# Hodgkin-Huxley neuron

A constant current makes the Hodgkin-Huxley neuron fire a regular train of action potentials: the sodium channels
open and the potential shoots up to about +40 mV, then the potassium channels open and pull it back below rest.
"""

# %%
# The model is created by name from the catalog with a current of 10 µA/cm², above the threshold of periodic firing,
# and run for 50 ms. The observer saves every second frame with the default blueprint, whose primary view is the time
# series of the potential and the gates.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hodgkin_huxley'

system = create(MODEL, I=10.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=50)

# %%
# Gates and the firing rate
# -------------------------
# During each spike the sodium activation `m` rises first, then the sodium inactivation `h` falls and the potassium
# activation `n` rises, which ends the spike. The spikes should be evenly spaced, with the period of about 14.6 ms
# Hodgkin and Huxley's neuron has at this current.

trajectory = simulation.trajectory_
spikes = np.array([event.time for event in trajectory.events if event.label == 'spike'])
fig, (potential, gates) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
potential.plot(trajectory.times, trajectory['neuron.v'])
potential.plot(spikes, np.full_like(spikes, 50.0), 'v', color='black', label='spike events')
potential.set_ylabel('potential (mV)')
potential.set_title(f'{spikes.size} spikes, mean interval {np.diff(spikes).mean():.1f} ms')
potential.legend()
for gate in ['m', 'h', 'n']:
    gates.plot(trajectory.times, trajectory[f'neuron.{gate}'], label=gate)
gates.set_xlabel('time (ms)')
gates.set_ylabel('gate opening')
gates.set_title('Sodium (m, h) and potassium (n) gates')
gates.legend()
fig.tight_layout()
plt.show()
