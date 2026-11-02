"""
# Amari neural field

A line of neurons excites its near neighbours and inhibits those further away, through a convolution of the firing rate
against a lateral-inhibition kernel. Below the firing threshold the field rests, but a local excitation settles into a
stationary bump of activity that outlasts its cause, a model of working memory. Amari's condition gives its width.
"""

# %%
# The model is created by name from the catalog on a line of length 20 in 800 sites, with the kernel of a Mexican hat,
# from a localized excitation. The default blueprint shows the history of the potential.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'amari_neural_field'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.5, observers=[observer]).check().prepare().run(until=40.0)

# %%
# Width of the bump
# -----------------
# The bump should stop changing, with its width at the root of `W(a) + h = 0` where the kernel is negative. With the
# defaults, `W(a) = a exp(-a)` and `h = -0.2`, and the width should be within 5% of the larger root.

potential = np.reshape(simulation.trajectory_.values('field.u'), (-1, 800))
positions = (np.arange(800) + 0.5) * 20 / 800
root = brentq(lambda a: a * np.exp(-a) - 0.2, 1.0, 10.0)
widths = (potential > 0).sum(axis=1) * 20 / 800

fig, (growth, profile) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(simulation.trajectory_.times, widths, label='width of the bump')
growth.axhline(root, color='black', linestyle='--', linewidth=0.8, label='root of W(a) + h')
growth.set_xlabel('time')
growth.set_ylabel('width')
growth.set_title('The bump settles at the predicted width')
growth.legend()
profile.plot(positions, potential[-1])
profile.axhline(0, color='black', linewidth=0.8)
profile.set_xlabel('position')
profile.set_ylabel('potential')
profile.set_title('A stationary bump above the threshold')
fig.tight_layout()
plt.show()
