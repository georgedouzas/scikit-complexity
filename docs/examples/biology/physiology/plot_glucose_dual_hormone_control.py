"""
# Glucose dual hormone control

Blood glucose is held near its norm by two hormones that act in opposite directions. A meal at hour 2 raises the
glucose, and insulin rises to bring it back down. An exercise at hour 12 lowers it, and glucagon rises to release
glucose and bring it back up. The view shows the glucose with its peak and its dip, the insulin that answers the
peak, and the glucagon that answers the dip.
"""

# %%
# The model is created by name from the catalog with both hormone loops, a meal of two hours from hour 2 and an
# exercise of two hours from hour 12. The glucose is relative to its norm, and the time unit is about an hour. The
# observer saves every second frame with the default blueprint, whose primary view is the time series of the glucose
# `G`, the insulin `I`, and the glucagon `N`.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'glucose_dual_hormone_control'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=24)

# %%
# Attenuation by two antagonistic loops
# -------------------------------------
# A receptor acts in one direction only, so insulin alone can only correct highs. Without the glucagon loop, the meal
# should be corrected almost as well, while the dip of the exercise should be deeper and last longer. With both loops
# the excursions to both sides should stay small, the attenuation of Konieczny et al.

single = Simulation(create(MODEL, glucagon=False), dt=0.05).run(until=24).trajectory_
times = simulation.trajectory_.times
fig, (glucose, hormones) = plt.subplots(1, 2, figsize=(10, 4))
glucose.plot(times, simulation.trajectory_['organism.G'], label='insulin and glucagon')
glucose.plot(times, single['organism.G'], label='insulin alone')
glucose.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
glucose.set_xlabel('time (hours)')
glucose.set_ylabel('glucose relative to the norm')
glucose.set_title('Glucagon limits the dip of the exercise')
glucose.legend()
hormones.plot(times, simulation.trajectory_['organism.I'], label='insulin')
hormones.plot(times, simulation.trajectory_['organism.N'], label='glucagon')
hormones.set_xlabel('time (hours)')
hormones.set_ylabel('hormone level')
hormones.set_title('Each hormone answers one side of the norm')
hormones.legend()
fig.tight_layout()
plt.show()
