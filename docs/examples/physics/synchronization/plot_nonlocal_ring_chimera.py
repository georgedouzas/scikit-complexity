"""
# Chimera on a ring

Identical phase oscillators on a ring, each coupled to its neighbours within a range, split into two coexisting
states: one stretch of the ring locks into a smooth, coherent wave while the rest keeps drifting incoherently. This
coexistence of order and disorder among identical units is a chimera state.
"""

# %%
# One hundred oscillators are coupled to their 14 nearest neighbours on each side with the phase lag
# `alpha = 1.46`. The model's blueprint shows the phase of every oscillator against its place on the ring, colored
# around the cycle of phases: the coherent stretch is a smooth curve, the incoherent one a scatter of points.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'nonlocal_ring_chimera'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=200)

# %%
# Frequencies along the ring
# --------------------------
# The mean frequency of every oscillator over the last 100 time units is the signature of a chimera: the coherent
# stretch should share one frequency, a plateau near -0.69, while the incoherent oscillators drift more slowly, at
# frequencies that bend away from the plateau in an arc, down to about -0.26.

phases = np.asarray(simulation.trajectory_['ring.theta'])
frequencies = (phases[-1] - phases[-101]) / 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(phases.shape[1]), frequencies, 'o-', markersize=3)
ax.set_xlabel('oscillator')
ax.set_ylabel('mean frequency')
ax.set_title('A locked plateau and a drifting arc coexist')
plt.show()
