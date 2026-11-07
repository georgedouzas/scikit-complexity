"""
# Noisy Boolean network

Every node of a random network follows the majority of its three inputs, but disobeys with a probability. Below a
critical noise the network orders, most nodes sharing one state, and above it the order melts into fluctuations, a
continuous phase transition that mean-field theory places at 1/6.
"""

# %%
# The system is created by name from the catalog with two hundred nodes at the noise 0.1, and the observer saves every
# step with the default blueprint, whose primary view is the time series of the states.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'noisy_boolean_network'

system = create_system(SYSTEM, n=200)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=200)

# %%
# The phase transition
# --------------------
# The order parameter `Psi`, the time average of `|mean sigma|`, should fall to zero near the mean-field critical
# noise `eta_c = 1/6`.

noises = np.linspace(0.0, 0.3, 13)
order = []
for eta in noises:
    run = Simulation(create_system(SYSTEM, eta=eta), random_state=0).prepare().run(until=150).trajectory_
    order.append(np.abs(run.values('nodes.sigma').mean(axis=1)[50:]).mean())
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(noises, order, 'o-')
ax.axvline(1 / 6, color='black', linestyle='--', label='eta_c = 1/6')
ax.set_xlabel('noise eta')
ax.set_ylabel('order parameter Psi')
ax.set_title('Order melts at the critical noise')
ax.legend()
plt.show()
