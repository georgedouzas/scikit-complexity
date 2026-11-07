"""
# Glycolysis oscillations

The glycolysis of a yeast extract fed with glucose in a flow reactor oscillates: NADH and every other metabolite rise
and fall with a period of minutes, driven by the feedback of ATP and AMP on phosphofructokinase.
"""

# %%
# The system is created by name from the catalog with the parameters of Nielsen et al. (1998), and the observer saves
# every third frame with the default blueprint, whose primary view is the time series of the metabolites.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'glycolysis_oscillations'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=3)
simulation = Simulation(system, dt=0.5, observers=[observer]).prepare().run(until=300)

# %%
# Effect of the dehydrogenase
# ---------------------------
# Raman (2021, exercise 5.7) changes `V4` by half. The oscillation of NADH should change in level and amplitude, while
# the nicotinamide pool `NAD + NADH` tends to the feed of 4.24 mM in every case.

fig, (nadh, pool) = plt.subplots(1, 2, figsize=(11, 4.5))
for v4 in (5.0, 10.0, 15.0):
    run = Simulation(create_system(SYSTEM, V4=v4), dt=0.5).prepare().run(until=300).trajectory_
    nadh.plot(run.times, run.values('extract.NADH'), label=f'V4 = {v4}')
    pool.plot(run.times, run.values('extract.NAD') + run.values('extract.NADH'), label=f'V4 = {v4}')
nadh.set_xlabel('time (min)')
nadh.set_ylabel('NADH (mM)')
nadh.set_title('NADH oscillations')
nadh.legend()
pool.axhline(4.24, color='black', linestyle='--')
pool.set_xlabel('time (min)')
pool.set_ylabel('NAD + NADH (mM)')
pool.set_title('Nicotinamide pool')
fig.tight_layout()
plt.show()
