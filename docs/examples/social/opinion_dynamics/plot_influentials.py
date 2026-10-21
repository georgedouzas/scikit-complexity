"""
# Influentials

A belief diffusion model with influentials: everyone on a grid weighs four levels of caring about an issue, from not
at all to passionately, and every step adds the influence of each neighbour to the weight of the level that neighbour
cares at. Beliefs lock in as the weights grow, so the grid freezes into patches of agreement. A handful of influentials
who care passionately and weigh more with their neighbours enlarge the passionate patches around them, and a news
report that nudges everyone towards passion sweeps almost the whole population.
"""

# %%
# Four hundred people fill a 20 by 20 torus with weights drawn uniformly on 0 to 4, and ten influentials weigh three
# times as much as the others. The blueprint shows how much everyone cares after 20 steps, from not at all (purple) to
# passionately (yellow), frozen into patches.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'influentials'
STEPS = 20
LEVELS = ('not at all', 'less', 'a bit', 'passionately')

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, influence=3.0, random_state=0), observers=[observer], random_state=0)
simulation.run(until=STEPS)

# %%
# Locking in, influence, and news
# -------------------------------
# The share of people at each level should change quickly at first and then freeze as the weights grow, with the
# passionate level gaining most because the influentials push it. Over three runs at each influence, the final share
# caring passionately should grow with the weight of the influentials, and a news report that adds one to everyone's
# passionate weight at the start should carry it close to the whole population.


def shares(trajectory, frame):
    """Return the share of people at each level of caring at a frame."""
    caring = trajectory.agents('people', frame).values['caring'].astype(int)
    return np.bincount(caring, minlength=len(LEVELS)) / len(caring)


def passionate(influence, news_report):
    """Return the mean final share of people caring passionately over three runs."""
    finals = []
    for seed in range(3):
        system = create(MODEL, influence=influence, news_report=news_report, random_state=seed)
        trajectory = Simulation(system, random_state=seed).run(until=STEPS).trajectory_
        finals.append(shares(trajectory, STEPS)[-1])
    return np.mean(finals)


trajectory = simulation.trajectory_
history = np.array([shares(trajectory, frame) for frame in range(STEPS + 1)])
influences = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
fig, (levels, sweep) = plt.subplots(1, 2, figsize=(10, 4))
for index, level in enumerate(LEVELS):
    levels.plot(trajectory.times, history[:, index], color=f'C{index}', label=level)
levels.set_xlabel('step')
levels.set_ylabel('share of people')
levels.set_title('Beliefs lock in within a few steps')
levels.legend(title='caring')
for news_report, color in ((False, 'C0'), (True, 'C1')):
    label = 'with a news report' if news_report else 'without a news report'
    sweep.plot(
        influences,
        [passionate(influence, news_report) for influence in influences],
        'o-',
        color=color,
        label=label,
    )
sweep.set_ylim(0, 1)
sweep.set_xscale('log')
sweep.set_xlabel('influence of the influentials')
sweep.set_ylabel('final share caring passionately')
sweep.set_title('Influentials and news spread passion')
sweep.legend()
fig.tight_layout()
plt.show()
