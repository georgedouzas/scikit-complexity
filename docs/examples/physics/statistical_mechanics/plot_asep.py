"""
# Asymmetric simple exclusion process

Particles enter an empty line of 100 sites at rate `alpha = 0.3` and leave at rate `gamma = 0.7`, so the line fills
to the low density `alpha` set by the entry, with a current `alpha (1 - alpha)`. The space-time image shows the
occupations over the last 100 frames, one row per frame with the earliest at the top: particles and small
clusters stream to the right along diagonal tracks, at low density.
"""

# %%
# The model is created by name from the catalog with forward hops at rate 1 and no backward hops. The observer saves
# every frame with the default blueprint, whose primary view is the space-time image of the occupations. The Numba
# backend runs the hops in a compiled loop.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'asep'

system = create(MODEL, n_sites=100, alpha=0.3, gamma=0.7)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=0, backend='numba').run(until=250)

# %%
# Density profiles and currents of the three phases
# -------------------------------------------------
# For `p = 1` and `q = 0` the phase diagram of Bertin's figure 3.6 has a low-density phase with bulk density `alpha`
# and current `alpha (1 - alpha)`, a high-density phase with bulk density `1 - gamma` and current `gamma (1 - gamma)`,
# and a maximal-current phase with bulk density `1/2` and current `1/4`. The time-averaged profiles should be flat at
# these densities away from the ends, and the measured current `gamma <n_L>` should match the predicted one.

phases = {'low density': (0.3, 0.7), 'high density': (0.7, 0.3), 'maximal current': (0.8, 0.8)}
fig, (profiles, currents) = plt.subplots(1, 2, figsize=(10, 4))
for index, (name, (alpha, gamma)) in enumerate(phases.items()):
    run = Simulation(create(MODEL, alpha=alpha, gamma=gamma), dt=2.0, random_state=0, backend='numba').run(until=4000)
    lane = run.trajectory_['lane.n'][250:]
    predicted = 0.25 if min(alpha, gamma) > 0.5 else min(alpha, gamma) * (1 - min(alpha, gamma))
    profiles.plot(lane.mean(axis=0), label=f'{name}, alpha = {alpha}, gamma = {gamma}')
    currents.bar(index - 0.2, gamma * lane[:, -1].mean(), width=0.4, color='tab:blue')
    currents.bar(index + 0.2, predicted, width=0.4, color='gray')
profiles.set_xlabel('site')
profiles.set_ylabel('mean occupation')
profiles.set_title('Density profiles of the three phases')
profiles.set_ylim(0, 1.1)
profiles.legend(loc='upper center', fontsize='small')
currents.set_xticks(range(3), list(phases))
currents.set_ylabel('current')
currents.set_title('Measured (blue) and predicted (gray) currents')
fig.tight_layout()
plt.show()
