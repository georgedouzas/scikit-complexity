"""
# SIS

An infection that leaves no immunity becomes endemic: the infected grow along an S-shaped curve and level off where
new infections balance recoveries, at `N - gamma / beta`. Every infection and recovery is a random event of one
individual, so the level fluctuates.
"""

# %%
# The model runs as a jump process from ten infected in a population of 1000. The observer saves every day with the
# default blueprint, whose primary view is the time series of both compartments.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sis'

system = create(MODEL, stochastic=True)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0).run(until=100)

# %%
# Logistic growth of the infected
# -------------------------------
# With `S = N - I`, the infected follow the logistic law `dI/dt = (beta N - gamma) I (1 - I / I*)` with
# `I* = N - gamma / beta`. The random run should follow the deterministic curve and fluctuate around `I*`.

trajectory = simulation.trajectory_
beta, gamma, total, i0 = 0.0003, 0.1, 1000.0, 10.0
rate, level = beta * total - gamma, total - gamma / beta
logistic = level / (1 + (level / i0 - 1) * np.exp(-rate * trajectory.times))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trajectory.times, trajectory['I.n'], label='infected, random run')
ax.plot(trajectory.times, logistic, '--', color='black', label='logistic solution')
ax.axhline(level, color='grey', linestyle=':', label=f'endemic level {level:.0f}')
ax.set_xlabel('time')
ax.set_ylabel('infected')
ax.set_title(f'Mean of the last 30 days: {trajectory["I.n"][-30:].mean():.0f} infected')
ax.legend()
plt.show()
