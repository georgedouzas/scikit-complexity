"""
# Adaptive delay network

Phase units are coupled through axons whose conduction delays adapt to the activity they carry, as myelination does.
Fixed delays leave the units incoherent, while delays that follow the phase differences of their units bring partial
synchrony back.
"""

# %%
# The example builds `AdaptiveDelayNetworkSystem` with five units, and the observer saves every other frame with the
# default blueprint, whose primary view is the time series of the phases and delays.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import AdaptiveDelayNetworkSystem

SYSTEM = 'adaptive_delay_network'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(AdaptiveDelayNetworkSystem(), observers=[observer])

# %%
# Synchrony with fixed and adaptive delays
# ----------------------------------------
# The order parameter `R = |mean exp(i theta)|` should stay low with fixed delays, `epsilon = 0`, and rise when the
# delays adapt.


def order(trajectory):
    phases = np.array([trajectory.get_values(f'unit_{index}.theta') for index in range(5)]).T
    return np.abs(np.exp(1j * phases).mean(axis=1))


fixed = simulate(AdaptiveDelayNetworkSystem(epsilon=0.0))
fig, (coherence, delays) = plt.subplots(1, 2, figsize=(11, 4.5))
coherence.plot(fixed.times, order(fixed), label='fixed delays')
coherence.plot(trajectory.times, order(trajectory), label='adaptive delays')
coherence.set_xlabel('time')
coherence.set_ylabel('order parameter R')
coherence.set_title('Adaptive delays restore synchrony')
coherence.legend()
for source in range(5):
    for target in range(5):
        if source != target:
            delays.plot(
                trajectory.times,
                trajectory.get_values(f'axon_{source}_{target}.tau'),
                lw=0.8,
            )
delays.set_xlabel('time')
delays.set_ylabel('conduction delay')
delays.set_title('Delays of the twenty axons')
fig.tight_layout()
plt.show()
