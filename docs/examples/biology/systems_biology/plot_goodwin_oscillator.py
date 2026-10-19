"""
# Goodwin oscillator

A product that represses the synthesis of its own mRNA through an enzyme oscillates when the repression is steep, with
a Hill coefficient of 12.
"""

# %%
# The model is created by name from the catalog with three stages, mRNA `x1`, enzyme `x2`, and product `x3`, and a
# Hill coefficient of 12. The observer saves every third frame of the 60 time units with the default blueprint, whose
# primary view is the time series of the three concentrations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'goodwin_oscillator'

system = create(MODEL, n=12.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=60)

# %%
# The threshold of the Hill coefficient
# -------------------------------------
# With three stages that decay at the same rate, the fixed point loses its stability when the loop gain exceeds
# `sec(pi / 3)^3 = 8`. With a strong maximal synthesis the gain is close to the Hill coefficient, so the late amplitude
# of the product should be zero below a Hill coefficient of about 8 and grow above it.

coefficients = np.arange(4.0, 15.0)
amplitudes = []
for n in coefficients:
    late = Simulation(create(MODEL, n=float(n)), dt=0.1).run(until=200).trajectory_
    product = late['x3.x'][late.times >= 150.0]
    amplitudes.append(product.max() - product.min())
trajectory = simulation.trajectory_
fig, (series, threshold) = plt.subplots(1, 2, figsize=(10, 4))
for name in ('x1', 'x2', 'x3'):
    series.plot(trajectory.times, trajectory[f'{name}.x'], label=name)
series.set_xlabel('time')
series.set_ylabel('concentration')
series.set_title('Sustained oscillations for n = 12')
series.legend()
threshold.plot(coefficients, amplitudes, 'o-')
threshold.axvline(8.0, color='grey', linestyle=':', label='secant threshold 8')
threshold.set_xlabel('Hill coefficient n')
threshold.set_ylabel('late amplitude of x3')
threshold.set_title('Oscillations appear above n = 8')
threshold.legend()
fig.tight_layout()
plt.show()
