"""
# Ricker

A population in discrete generations whose crowded generations leave fewer offspring settles on a fixed point at low
growth, and at high growth overshoots and crashes in cycles that double their period on the way to chaos.
"""

# %%
# The system is built from its class with the growth factor `r = 14`, in the chaotic range, and the observer saves
# every generation with the default blueprint, whose primary view is the time series of the population.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import RickerSystem

SYSTEM = 'ricker'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(RickerSystem(r=14.0), observers=[observer])

# %%
# Period doubling
# ---------------
# The fixed point `n = ln r` is stable up to `r = e^2`, about 7.39, where it gives way to a two-cycle. The late
# generations of runs over a range of `r` should trace the period-doubling cascade, and the run above should lie in
# its chaotic band.

rates = np.linspace(1.5, 20.0, 150)
fig, (series, diagram) = plt.subplots(1, 2, figsize=(11, 4.5))
series.plot(trajectory.times, trajectory.values('population.n'), '.-')
series.set_xlabel('generation')
series.set_ylabel('population n')
series.set_title('r = 14')
for r in rates:
    late = simulate(RickerSystem(r=r), until=200).values('population.n')[-60:]
    diagram.plot(np.full(late.size, r), late, ',', color='black')
diagram.plot(rates[rates < np.e**2], np.log(rates[rates < np.e**2]), color='tab:red', label='fixed point ln r')
diagram.axvline(np.e**2, color='gray', linestyle='--', label='r = e^2')
diagram.set_xlabel('growth factor r')
diagram.set_ylabel('late generations')
diagram.set_title('Bifurcation diagram')
diagram.legend()
fig.tight_layout()
plt.show()
