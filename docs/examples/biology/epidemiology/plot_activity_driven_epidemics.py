"""
# Activity-driven epidemics

On an activity-driven network every edge lasts one step: active nodes join a few random others and the edges vanish
at the next step. The infection travels along these short-lived edges and settles at an endemic level when the ratio
of infection and recovery exceeds a threshold set by the activities, not by the network aggregated over time.
"""

# %%
# The example builds `ActivityDrivenEpidemicsSystem` with 500 nodes whose activities follow a power law. The observer
# saves every second step with the default blueprint, whose primary view is the network of the current step, with the
# infected nodes bright.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.epidemiology import ActivityDrivenEpidemicsSystem

SYSTEM = 'activity_driven_epidemics'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(ActivityDrivenEpidemicsSystem(), observers=[observer], random_state=1)

# %%
# Prevalence across the threshold
# -------------------------------
# Sweeping the infection probability at a fixed recovery probability, the late prevalence should stay at zero for
# small probabilities and rise beyond a threshold. The edges of every step number about `2 m <a> n`.

edges = np.mean([trajectory.get_network(frame).sources.size for frame in range(1, len(trajectory.times))])
activity = ActivityDrivenEpidemicsSystem().prepare().system_.elements[0].values['a']
probabilities = np.linspace(0.05, 0.9, 8)
prevalence = []
for lam in probabilities:
    run = simulate(ActivityDrivenEpidemicsSystem(lam=lam), dt=10.0, random_state=1)
    prevalence.append(np.nanmean(run.get_values('hosts.i')[-5:]))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(probabilities / 0.1, prevalence, 'o-')
ax.set_xlabel('lam / mu')
ax.set_ylabel('late prevalence')
ax.set_title(f'{edges:.0f} edges per step, 2 m <a> n = {4 * activity.mean() * activity.size:.0f}')
plt.show()
