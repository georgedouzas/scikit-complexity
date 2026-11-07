"""
# Two-layer sepsis

Parenchymal and immune cells are two layers of phase oscillators whose couplings, the cytokines, adapt to their phase
differences. A low age parameter lets both layers lock into the healthy, frequency-synchronized state, while a higher
one leaves them in a desynchronized, pathological state.
"""

# %%
# The system is created by name from the catalog with 20 cells per layer at the age parameter `beta = pi / 2`. The
# observer saves every frame with the default blueprint, whose primary view is the network of cells with their phases.

import math

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'two_layer_sepsis'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer]).prepare().run(until=200)

# %%
# Healthy and pathological states
# -------------------------------
# The spread of the mean frequencies of the parenchymal cells should vanish in the healthy state, at a low `beta`, and
# stay large in the pathological one.

betas = np.linspace(0.2, 0.6, 5) * math.pi
spreads = []
for beta in betas:
    run = Simulation(create_system(SYSTEM, beta=beta), dt=1.0).prepare().run(until=200).trajectory_
    phases = np.unwrap(run.values('cells.phi'), axis=0)
    spreads.append(((phases[-1] - phases[-51]) / 50)[:20].std())
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(betas / math.pi, spreads, 'o-')
ax.set_xlabel('age parameter beta / pi')
ax.set_ylabel('spread of the parenchymal frequencies')
ax.set_title('The age parameter desynchronizes the tissue')
plt.show()
