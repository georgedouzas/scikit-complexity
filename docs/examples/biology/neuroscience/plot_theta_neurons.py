"""
# Theta neurons

Fifty theta neurons that are each below their firing threshold stay quiet on their own. Coupled strongly enough by
excitatory pulses, they kick each other over threshold and the whole network fires together in collective bursts.
"""

# %%
# The system is built from its class with excitabilities centred below threshold, `eta = -0.2`, and the coupling
# `kappa = 1.5`. The recorded run keeps four neurons for ten time units, so that their phases read in the time series
# view of the default blueprint. The analysis below uses fifty.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import ThetaNeuronsSystem

SYSTEM = 'theta_neurons'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(ThetaNeuronsSystem(n_neurons=4, kappa=1.5, random_state=0), observers=[observer], until=10)

# %%
# Coupling switches on collective firing
# --------------------------------------
# A neuron fires when its phase crosses `π`. Marking every crossing, the uncoupled network should stay almost silent,
# with only the few neurons of the Lorentzian tail above threshold firing, while the coupled network should fire many
# times more often, in volleys shared by many neurons.


def spikes_of(trajectory):
    """Return the step and the neuron of every crossing of π."""
    wrapped = np.mod(trajectory.values('neurons.theta'), 2 * np.pi)
    steps, neurons = np.nonzero((wrapped[:-1] < np.pi) & (wrapped[1:] >= np.pi))
    return trajectory.times[steps + 1], neurons


coupled = simulate(ThetaNeuronsSystem(n_neurons=50, kappa=1.5, random_state=0))
uncoupled = simulate(ThetaNeuronsSystem(n_neurons=50, kappa=0.0, random_state=0))
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
