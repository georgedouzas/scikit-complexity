"""
# Generalized logistic growth

Tumours with three exponents of saturation grow along S-shaped curves to the same carrying capacity, and the larger
exponent keeps the growth fast for longer before it stops.
"""

# %%
# The model is created by name from the catalog with the logistic fit of Padmanabhan et al. to breast cancer,
# `r = 0.305` and `K = 1221`, and the exponent `p = 2`, and run for 40 days from 10 mm³. The observer saves every
# frame with the default blueprint, whose primary view is the time series of the volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'generalized_logistic'

system = create(MODEL, n=10.0, p=2.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=40)

# %%
# Exponents of the saturation
# ---------------------------
# Runs with `p = 0.5`, 1, and 2 should each follow the closed form
# `n0 K / (n0^p + (K^p - n0^p) exp(-r p t))^(1/p)`, and the per capita rate `(dn/dt) / n` should fall from `r` to 0 as
# `1 - (n / K)^p`.

r, K, n0 = 0.305, 1221.0, 10.0
fig, (curves, rates) = plt.subplots(1, 2, figsize=(10, 4))
for p in (0.5, 1.0, 2.0):
    trajectory = Simulation(create(MODEL, n=n0, p=p), dt=0.2).run(until=40).trajectory_
    times, volume = trajectory.times, trajectory['tumour.n']
    exact = n0 * K / (n0**p + (K**p - n0**p) * np.exp(-r * p * times)) ** (1 / p)
    curves.plot(times, volume, label=f'p = {p}')
    curves.plot(times, exact, '--', color='black', linewidth=0.8)
    rates.plot(volume / K, np.gradient(volume, times) / volume / r, label=f'p = {p}')
curves.set_xlabel('time (days)')
curves.set_ylabel('volume (mm³)')
curves.set_title('Simulations on their closed forms (dashed)')
curves.legend()
rates.set_xlabel('volume / K')
rates.set_ylabel('per capita rate / r')
rates.set_title('The per capita rate falls as 1 - (n / K)^p')
rates.legend()
fig.tight_layout()
plt.show()
