"""
# Hutchinson's delayed logistic

A population that responds to its density with a delay overshoots its carrying capacity, crashes below it, and
settles on sustained oscillations around it, because the delay `tau` times the growth rate `r` exceeds π/2.
"""

# %%
# The model is created by name from the catalog with `r tau = 2`, above the threshold of π/2. The observer saves
# every fourth frame with the default blueprint, whose primary view is the time series of the population.

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hutchinson'

system = create(MODEL, n=50.0, r=1.0, K=100.0, tau=2.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=100)

# %%
# Period of the oscillation
# -------------------------
# Near the threshold the period of the oscillation is `4 tau`, and it lengthens as `r tau` grows. The time between
# successive peaks of the settled oscillation should be somewhat above `4 tau = 8`, and the same at every peak.

times, population = simulation.trajectory_.times, simulation.trajectory_['population.n']
peaks, _ = find_peaks(population)
periods = np.diff(times[peaks])
fig, (series, period) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(times, population)
series.plot(times[peaks], population[peaks], 'o', color='black', label='peaks')
series.axhline(100.0, color='grey', linestyle=':', label='carrying capacity')
series.set_xlabel('time')
series.set_ylabel('population')
series.set_title('Sustained oscillations around K')
series.legend()
period.plot(times[peaks][1:], periods, 'o-', label='time between peaks')
period.axhline(8.0, color='black', linestyle='--', label='4 tau, the period at the threshold')
period.set_ylim(0, 1.5 * periods.max())
period.set_xlabel('time')
period.set_ylabel('period')
period.set_title(f'Period settles at {periods[-1]:.2f}')
period.legend()
fig.tight_layout()
plt.show()
