"""
# Obesity

Adults eat three meals a day at restaurants near home and work, with portions set by their eating habit. Over one
year the low-habit eaters slim down while the medium and high-habit eaters gain weight in step with their habit, and a
growing share of them crosses the obesity line at a body mass index of 30.
"""

# %%
# The model is created by name from the catalog with 100 people, 20 homes, 20 workplaces, and 20 restaurants of three
# kinds on a square of side 280, and runs for one year, one step per day. A person eats at a random restaurant within
# reach, or at the nearest one, and spends 1.55 times the basal metabolic rate a day. The default blueprint shows the
# people at home, colored by body mass index from 18.5 (dark) to 35 (bright), next to the larger restaurants: the colors
# spread as the habits pull the body mass indices apart.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'obesity'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=364)

# %%
# Body mass index by habit
# ------------------------
# The weight changes once a week by the calories eaten beyond the week's expenditure. A low-habit meal averages about
# 740 kilocalories over the three kinds of restaurant, a medium one about 880, and a high one about 1030. Against an
# expenditure near 2500 kilocalories a day, three meals leave the low-habit group about 250 kilocalories short and the
# medium and high groups about 150 and 600 over, so the low group should lose weight and the high group gain fastest.
# The gains slow down as heavier bodies burn more, and the quartiles stay wide because the nearby kinds of restaurant
# differ: a low-habit eater who can only reach fast food still gains.

trajectory = simulation.trajectory_
frames = range(0, len(trajectory.times), 7)
days = trajectory.times[list(frames)]
habits = np.asarray(trajectory.agents('people', 0).values['habit'])
bmi = np.array([np.asarray(trajectory.agents('people', frame).values['bmi']) for frame in frames])
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))
for habit, name in enumerate(('low', 'medium', 'high')):
    group = bmi[:, habits == habit]
    left.plot(days, group.mean(axis=1), label=f'{name} habit')
    left.fill_between(days, *np.percentile(group, [25, 75], axis=1), alpha=0.2)
    right.plot(days, (group >= 30).mean(axis=1), label=f'{name} habit')
left.axhline(30, color='gray', linestyle='--', linewidth=1)
left.set_xlabel('day')
left.set_ylabel('body mass index (mean and quartiles)')
left.set_title('Habits pull the body mass index apart')
left.legend()
right.set_xlabel('day')
right.set_ylabel('share with body mass index of 30 or more')
right.set_title('Obesity spreads in the heavy-eating groups')
plt.show()
