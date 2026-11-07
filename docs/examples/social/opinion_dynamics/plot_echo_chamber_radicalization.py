"""
# Echo chambers and radicalization

Baumann and coauthors couple opinions to an activity-driven network that is redrawn all the time. Agents push the
opinions of those they contact toward their own sign, more so the more convinced they are, and they prefer contacts of
similar opinion. A controversial issue radicalizes the population, to one side without homophily and into two
opposite echo chambers with it.
"""

# %%
# Two hundred agents with power-law activities debate an issue of controversialness `alpha = 3` with homophily
# `beta = 3`, for ten time units. The default blueprint shows the opinions.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.opinion_dynamics import EchoChamberRadicalizationSystem

SYSTEM = 'echo_chamber_radicalization'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(EchoChamberRadicalizationSystem(), observers=[observer], random_state=0)

# %%
# Consensus, radicalization, and polarization
# -------------------------------------------
# As in Fig. 1 of the paper, opinions should converge to zero for `alpha = 0.05`, all move to one side for `alpha = 3`
# without homophily, and split into two radical groups with homophily.

cases = {
    'consensus, alpha = 0.05': EchoChamberRadicalizationSystem(controversy=0.05, homophily=2.0),
    'radicalization, beta = 0': EchoChamberRadicalizationSystem(homophily=0.0),
    'polarization, beta = 3': EchoChamberRadicalizationSystem(),
}
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, (name, system) in zip(axes, cases.items(), strict=True):
    trajectory = simulate(system, until=100, random_state=0)
    opinions = np.array([trajectory.agents('agents', frame).values['opinion'] for frame in range(101)])
    axis.plot(np.asarray(trajectory.times) / 10, opinions[:, ::5], linewidth=0.6)
    axis.set_xlabel('time')
    axis.set_title(name)
axes[0].set_ylabel('opinion')
fig.tight_layout()
plt.show()
