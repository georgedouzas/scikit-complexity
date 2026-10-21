"""
# Heterogeneous mean-field SIS

On a network with a broad degree distribution the hubs catch the infection first and keep it alive: the infected
density of every degree class settles at an endemic level that rises with the degree.
"""

# %%
# The model is created by name from the catalog with degrees from 3 to 30 distributed as `k^-2.5` and a spreading
# rate above the threshold. The observer saves every frame with the default blueprint, whose primary view is the time
# series of the densities of the degree classes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'heterogeneous_mean_field_sis'

system = create(MODEL, lam=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=50)

# %%
# Endemic density by degree
# -------------------------
# At the endemic state `rho_k = lam k Theta / (1 + lam k Theta)`, with `Theta` the probability that a link points to
# an infected node. Fitting the one number `Theta` to the final densities, the simulated densities should fall on that
# curve, and the threshold `<k> / <k^2>` should lie below `lam`.

degrees = np.arange(3, 31)
weights = degrees**-2.5 / np.sum(degrees**-2.5)
threshold = np.sum(weights * degrees) / np.sum(weights * degrees**2)
final = simulation.trajectory_['network.rho'][-1]
theta = np.sum(weights * degrees * final) / np.sum(weights * degrees)
lam = 0.2
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(degrees, final, 'o', label='simulated')
ax.plot(degrees, lam * degrees * theta / (1 + lam * degrees * theta), '--', color='black', label='mean-field law')
ax.set_xlabel('degree k')
ax.set_ylabel('infected density rho_k')
ax.set_title(f'Hubs are infected most; threshold {threshold:.3f} < lam = {lam}')
ax.legend()
plt.show()
