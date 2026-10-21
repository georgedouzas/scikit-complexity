"""
# Theta neurons

Fifty theta neurons that are each below their firing threshold stay quiet on their own. Coupled strongly enough by
excitatory pulses, they kick each other over threshold and the whole network fires together in collective bursts.
"""

# %%
# The model is created by name from the catalog with excitabilities centred below threshold, `eta = -0.2`, and the
# coupling `kappa = 1.5`. The recorded run keeps eight neurons, so that their phases read in the time series view of
# the default blueprint; the analysis below uses fifty.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'theta_neurons'

system = create(MODEL, n_neurons=8, kappa=1.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=20)

# %%
# Coupling switches on collective firing
# --------------------------------------
# A neuron fires when its phase crosses `π`. Marking every crossing, the uncoupled network should stay almost silent,
# with only the few neurons of the Lorentzian tail above threshold firing, while the coupled network should fire many
# times more often, in volleys shared by many neurons.


def spikes_of(trajectory):
    """Return the step and the neuron of every crossing of π."""
    wrapped = np.mod(trajectory['neurons.theta'], 2 * np.pi)
    steps, neurons = np.nonzero((wrapped[:-1] < np.pi) & (wrapped[1:] >= np.pi))
    return trajectory.times[steps + 1], neurons


coupled = Simulation(create(MODEL, n_neurons=50, kappa=1.5, random_state=0), dt=0.1).run(until=20).trajectory_
uncoupled = Simulation(create(MODEL, n_neurons=50, kappa=0.0, random_state=0), dt=0.1).run(until=20).trajectory_
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, trajectory, label in zip(axes, [uncoupled, coupled], ['kappa = 0', 'kappa = 1.5'], strict=True):
    times, neurons = spikes_of(trajectory)
    ax.plot(times, neurons, '|', color='black')
    ax.set_xlabel('time')
    ax.set_title(f'{label}: {times.size} spikes')
axes[0].set_ylabel('neuron')
fig.suptitle('Coupling turns a quiet network into collective firing')
fig.tight_layout()
plt.show()
