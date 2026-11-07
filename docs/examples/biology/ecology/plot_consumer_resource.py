"""
# Consumer-resource

Two consumers that never meet compete only through the two resources they eat. They settle where neither can grow any
more, the crossing of their zero net growth isoclines, and their coexistence lasts because each eats relatively more of
the resource that limits it most.
"""

# %%
# The system is created by name from the catalog with the settings of Mehta and Blumenthal (2024, figure 4), and the
# observer saves every frame with the default blueprint, whose primary view is the time series of the consumers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'consumer_resource'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).prepare().run(until=100)

# %%
# Isoclines in resource space
# ---------------------------
# A consumer stops growing on its isocline `sum_b c_ib R_b = m_i`. The resources should travel from the supply point
# `K` to the crossing of both isoclines at `R = (5/3, 5/9)`.

resources = simulation.trajectory_.values('resources.R')
uptake = np.array([[0.5, 0.3], [0.4, 0.6]])
first = np.linspace(0, 5, 50)
fig, ax = plt.subplots(figsize=(6, 5))
for row, colour in zip(uptake, ('tab:blue', 'tab:orange'), strict=True):
    ax.plot(first, (1.0 - row[0] * first) / row[1], color=colour, label=f'isocline of consumer with c = {row}')
ax.plot(resources[:, 0], resources[:, 1], color='black', label='resources')
ax.plot(4.8, 2.85, 's', color='gray', label='supply point K')
ax.plot(5 / 3, 5 / 9, '*', color='red', markersize=12, label='coexistence')
ax.set_xlim(0, 5)
ax.set_ylim(0, 3.5)
ax.set_xlabel('resource 1')
ax.set_ylabel('resource 2')
ax.set_title('Resources settle where the isoclines cross')
ax.legend(fontsize=8)
plt.show()
