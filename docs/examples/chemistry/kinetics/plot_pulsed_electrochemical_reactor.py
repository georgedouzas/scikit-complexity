"""
# Pulsed electrochemical reactor

Every current pulse charges the electrode and raises its overpotential `eta`, which drives a reaction that consumes
the reactant `c`. Between pulses the overpotential relaxes and the feed restores the reactant, so the reactor settles
into a periodic response locked to the pulse train.
"""

# %%
# The model is created by name from the catalog with pulses of current 5 lasting 2 time units every 10 time units, and
# runs for twelve pulses. The default blueprint shows the overpotential and the reactant in one time series view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'pulsed_electrochemical_reactor'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=5)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=120)

# %%
# Locking to the pulse train
# --------------------------
# Cut into one piece per pulse period, the response should repeat: after the first few pulses the pieces of `eta` and
# `c` should lie on top of each other, and the reactant left at the start of each pulse should converge to one value.

trajectory = simulation.trajectory_
period = 10.0
samples = round(period / 0.1)
eta, c = trajectory['electrode.eta'][:-1], trajectory['electrode.c'][:-1]
phase = np.arange(samples) * 0.1
fig, (cycles, strobe) = plt.subplots(1, 2, figsize=(10, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(eta) // samples))
for index, color in enumerate(colors):
    piece = slice(index * samples, (index + 1) * samples)
    cycles.plot(phase, eta[piece], color=color, linewidth=1)
    cycles.plot(phase, c[piece], color=color, linewidth=1, linestyle='--')
cycles.set_xlabel('time since the pulse started')
cycles.set_ylabel('eta (solid) and c (dashed)')
cycles.set_title('Successive periods, from purple to yellow')
strobe.plot(np.arange(len(colors)), c[::samples][: len(colors)], 'o-')
strobe.set_xlabel('pulse')
strobe.set_ylabel('reactant c at the start of the pulse')
strobe.set_title('The stroboscopic reactant converges')
fig.tight_layout()
plt.show()
