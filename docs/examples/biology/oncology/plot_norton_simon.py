"""
# Norton-Simon Gompertzian cell kill

A Gompertzian tumour approaches its capacity until chemotherapy starts on day 42, and then every dose shrinks it and it
regrows quickly, since a smaller Gompertzian tumour grows faster.
"""

# %%
# The model is created by name from the catalog: a tumour of 100 mm³ grows towards 1964 mm³, and from day 42 a one-day
# dose every three weeks kills at the rate 1. The observer saves every second frame with the default blueprint, whose
# primary view is the time series of the volume.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'norton_simon'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=150)

# %%
# Regrowth after a dose
# ---------------------
# Between doses the per capita growth rate is `r log(K / n)`, which is larger the smaller the tumour. The rate measured
# from the simulation should lie on this line, and the doses should show as dips below it.

times, volume = simulation.trajectory_.times, simulation.trajectory_['tumour.n']
r, K = 0.0719, 1964.0
rate = np.gradient(np.log(volume), times)
fig, (curve, kinetics) = plt.subplots(1, 2, figsize=(10, 4))
curve.plot(times, volume)
curve.axhline(K, color='grey', linestyle=':', label='carrying capacity')
curve.set_xlabel('time (days)')
curve.set_ylabel('volume (mm³)')
curve.set_title('Doses every three weeks from day 42')
curve.legend()
kinetics.plot(volume, rate, '.', markersize=3, label='simulated')
grid = np.linspace(volume.min(), K, 100)
kinetics.plot(grid, r * np.log(K / grid), '--', color='black', label='r log(K / n)')
kinetics.set_xlabel('volume (mm³)')
kinetics.set_ylabel('per capita growth rate (1 / day)')
kinetics.set_title('Smaller tumours regrow faster')
kinetics.legend()
fig.tight_layout()
plt.show()
