"""
# Two-compartment pharmacokinetics

After a bolus of doxorubicin the plasma concentration falls fast while the drug spreads into the tissues, and then
slowly as the body eliminates it, while the tissue concentration rises and follows it down.
"""

# %%
# The model is created by name from the catalog with the rates of doxorubicin and a unit bolus, and runs for 3000
# minutes. The observer saves every second frame with the default blueprint, whose primary view is the time series of
# the central and peripheral concentrations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'two_compartment_pharmacokinetics'

system = create(MODEL, c0=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=10.0, observers=[observer]).run(until=3000)

# %%
# Bi-exponential decay
# --------------------
# The central concentration is `c0 ((alpha - k21) e^(-alpha t) + (k21 - beta) e^(-beta t)) / (alpha - beta)`, where
# `alpha` and `beta` are the roots of `x^2 - (k10 + k12 + k21) x + k10 k21`. On a logarithmic scale it should show two
# straight phases, of slopes `-alpha` and `-beta`, and lie on the closed form.

k10, k12, k21 = 0.082, 0.0475, 0.00125
total = k10 + k12 + k21
root = np.sqrt(total**2 - 4 * k10 * k21)
alpha, beta = (total + root) / 2, (total - root) / 2
times = simulation.trajectory_.times
exact = ((alpha - k21) * np.exp(-alpha * times) + (k21 - beta) * np.exp(-beta * times)) / (alpha - beta)
fig, (central, drug) = plt.subplots(1, 2, figsize=(10, 4))
central.semilogy(times, simulation.trajectory_['patient.Uc'], label='simulated')
central.semilogy(times, exact, '--', color='black', label='bi-exponential')
central.semilogy(times, (k21 - beta) / (alpha - beta) * np.exp(-beta * times), ':', label='slow phase')
central.set_ylim(1e-5, 1.5)
central.set_xlabel('time (min)')
central.set_ylabel('central concentration')
central.set_title('Distribution then elimination')
central.legend()
drug.plot(times, simulation.trajectory_['patient.Uc'], label='central')
drug.plot(times, simulation.trajectory_['patient.Up'], label='peripheral')
drug.plot(times, simulation.trajectory_['patient.Uc'] + simulation.trajectory_['patient.Up'], '--', label='total')
drug.set_xlabel('time (min)')
drug.set_ylabel('concentration')
drug.set_title('Only elimination lowers the total')
drug.legend()
fig.tight_layout()
plt.show()
