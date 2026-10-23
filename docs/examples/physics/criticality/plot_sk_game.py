"""
# SK game

Reinforcement learners on a spin-glass landscape each choose an action of ±1 and earn the reward their couplings with
the others give it. At a low exploration temperature they lock into one of many frustrated but rewarding
configurations, and learning stops. The view shows the mean reward per agent, which climbs from zero for random
actions to a plateau once the agents freeze.
"""

# %%
# The model is created by name from the catalog with 256 agents, the memory `alpha = 0.5`, and the temperature
# `T = 0.2`. Every agent starts with a zero incentive and a random action. The blueprint shows the standard observable
# `reward` over the whole run of 200 steps.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'sk_game'

# The view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/reward', name='mean reward per agent', axis_x=whole_run),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, temperature=0.2)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    dt=1,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(200)

# %%
# Freezing and exploration
# ------------------------
# The temperature sets the exploration of the softmax policy. At low temperature the agents should freeze, so the
# fraction of agents that change their action per step falls to zero while the reward stays high. At high temperature
# they should flip often and earn little, since random actions earn nothing on average (Lesmana et al., Sec. II.A).

temperatures = np.geomspace(0.05, 5.0, 12)
rewards, flips = [], []
for temperature in temperatures:
    run = create(MODEL, temperature=temperature)
    couplings = run.elements[0].values['couplings']
    actions = np.asarray(Simulation(run, dt=1, random_state=0).run(150).trajectory_['agents.sigma'])[-50:]
    rewards.append(np.einsum('ti,ij,tj->t', actions, couplings, actions).mean() / actions.shape[1])
    flips.append((actions[1:] != actions[:-1]).mean())

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(temperatures, rewards, 'o-', label='mean reward per agent')
ax.semilogx(temperatures, flips, 's--', label='fraction of agents flipping per step')
ax.set_xlabel('temperature T')
ax.set_ylabel('late-time average')
ax.set_title('Low temperature freezes a rewarding state')
ax.legend()
fig.tight_layout()
plt.show()
