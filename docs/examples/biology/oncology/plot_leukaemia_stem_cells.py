"""
# Leukaemic and normal stem cells

A single leukaemic stem cell, which renews a little faster, grows in the niche and drives the normal stem cells and
their progeny out of it.
"""

# %%
# The system is created by name from the catalog with a niche of 100 normal stem cells and one leukaemic stem cell, and
# runs for 500 time units. The observer saves every second frame with the default blueprint, whose primary view is the
# time series of the five populations.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'leukaemia_stem_cells'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).prepare().run(until=500)

# %%
# Competitive exclusion in the niche
# ----------------------------------
# Both kinds of stem cells share the feedback of the niche, so the kind with the larger net renewal excludes the
# other. As the renewal rate `r4` of the leukaemic stem cells crosses the normal rate, the final leukaemic share of the
# niche should switch from zero to one.

trajectory = simulation.trajectory_
fig, (series, switch) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times, trajectory.values('niche.NS'), label='normal stem cells')
series.plot(trajectory.times, trajectory.values('niche.AS'), label='leukaemic stem cells')
series.set_xlabel('time')
series.set_ylabel('cells')
series.set_title('The leukaemic stem cells take over')
series.legend()
renewals = [0.45, 0.5, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7, 0.8]
shares = []
for r4 in renewals:
    run = Simulation(create_system(SYSTEM, r4=r4), dt=5.0).prepare().run(until=3000).trajectory_
    shares.append(run.values('niche.AS')[-1] / (run.values('niche.AS')[-1] + run.values('niche.NS')[-1]))
switch.plot(renewals, shares, 'o-')
switch.axvline(0.6, color='grey', linestyle=':', label='normal renewal r1')
switch.set_xlabel('leukaemic renewal rate r4')
switch.set_ylabel('final leukaemic share of the niche')
switch.set_title('The fitter stem cells exclude the others')
switch.legend()
fig.tight_layout()
plt.show()
