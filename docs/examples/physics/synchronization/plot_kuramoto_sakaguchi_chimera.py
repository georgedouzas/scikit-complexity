"""
# Kuramoto-Sakaguchi chimera

Two populations of identical phase oscillators, coupled more strongly within than between them and with a phase lag
close to a quarter turn, split into a chimera: one population locks into perfect synchrony while the other keeps
drifting, its order rising and falling in a slow breathing rhythm.
"""

# %%
# Each population has 32 oscillators. The first starts nearly synchronized and the second spread over the circle, but
# for these initial phases it is the second that locks and the first that drifts. The system's blueprint shows the
# phases of each population against the index of the oscillator, colored around the cycle of phases, the first
# population first: a loose cluster of one color with stragglers spread over the circle.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.synchronization import KuramotoSakaguchiChimeraSystem

SYSTEM = 'kuramoto_sakaguchi_chimera'

system = KuramotoSakaguchiChimeraSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(system, until=300, observers=[observer])

# %%
# Order parameters
# ----------------
# The order parameter `|<exp(i θ)>|` of the synchronized population should stay at 1, while that of the drifting one
# should oscillate well below it, between about 0.58 and 0.83 here, the breathing chimera that Abrams and coworkers
# found for `A = 0.2`.


def order(population):
    """Return the order parameter of a population at every frame."""
    phases = np.asarray(trajectory.get_values(f'{population}.theta'))
    return np.abs(np.exp(1j * phases).mean(axis=1))


times = trajectory.times
fig, ax = plt.subplots(figsize=(7, 4))
for population in ('first', 'second'):
    ax.plot(times, order(population), label=f'{population} population')
ax.set_ylim(0, 1.05)
ax.set_xlabel('time')
ax.set_ylabel('order parameter')
ax.set_title('One population locks while the other breathes')
ax.legend()
plt.show()
