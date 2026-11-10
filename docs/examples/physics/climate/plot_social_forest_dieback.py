"""
# Forest dieback with replicator dynamics of mitigators

A population decides whether to mitigate climate change by weighing the cost of mitigation against the warming it
perceived over the last ten years, and its mitigation helps a forest that warms the climate when it dies back. The two
feed back on each other with a delay.
"""

# %%
# The forest starts at the cover 0.8 and half of the population mitigates, at the ambient temperature of 30 degrees and
# the critical warming `Tc = 1`. The default blueprint shows the cover and the share of mitigators.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.climate import SocialForestDiebackSystem

SYSTEM = 'social_forest_dieback'

system = SocialForestDiebackSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(system, observers=[observer])

# %%
# Outcomes after 200 years
# ------------------------
# For several ambient temperatures and critical warmings, the population should end with all or no mitigators, and the
# forest with a positive cover or bare soil. Sensitive populations, with a low critical warming, should mitigate and
# keep their forest over a wider range of temperatures.

temperatures = np.linspace(26.0, 34.0, 17)
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for tc in (0.25, 1.0, 2.0):
    covers, shares = [], []
    for t in temperatures:
        run = simulate(SocialForestDiebackSystem(Tv=t, Tc=tc), dt=1.0)
        covers.append(np.ravel(run.get_values('forest.v'))[-1])
        shares.append(np.ravel(run.get_values('people.x'))[-1])
    axes[0].plot(temperatures, covers, 'o-', label=f'Tc = {tc}')
    axes[1].plot(temperatures, shares, 'o-', label=f'Tc = {tc}')
axes[0].set_ylabel('final value')
axes[0].set_title('forest cover')
axes[1].set_title('share of mitigators')
for ax in axes:
    ax.set_xlabel('ambient temperature (degrees Celsius)')
axes[1].legend()
plt.show()
