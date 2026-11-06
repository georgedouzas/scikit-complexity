"""
# Record-dynamics epidemic

A yearly strain infects every resident whose resistance is below its strength, and survivors of an infection are immune
for good. An outbreak needs a strain stronger than every earlier one, a record, so outbreaks come ever more rarely, at
a constant rate in the logarithm of time.
"""

# %%
# The system is created by name from the catalog with 2000 residents and run for 300 years. The observer saves every
# other year with the default blueprint, whose primary view is the time series of the strength of the strain.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'record_dynamics_epidemic'

system = create_system(SYSTEM, n=2000)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=300)

# %%
# Outbreaks in logarithmic time
# -----------------------------
# The mean number of records among `t` draws is the harmonic number `H_t`, about `ln t + 0.577`. The cumulative number
# of outbreaks, averaged over runs, should follow it.

years = np.arange(1, 301)
counts = []
for seed in range(20):
    run = Simulation(create_system(SYSTEM, n=2000, seed=seed), random_state=seed).prepare().run(until=300)
    counts.append(np.cumsum(run.trajectory_.values('residents.infected').sum(axis=1)[1:] > 0))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(years, np.mean(counts, axis=0), label='mean of 20 runs')
ax.plot(years, np.cumsum(1 / years) - 1 / years, '--', color='black', label='harmonic number of the strains drawn')
ax.set_xscale('log')
ax.set_xlabel('year')
ax.set_ylabel('cumulative outbreaks')
ax.set_title('Outbreaks follow the records of the strain strength')
ax.legend()
plt.show()
