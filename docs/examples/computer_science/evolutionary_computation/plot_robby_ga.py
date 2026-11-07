"""
# Robby the robot

A genetic algorithm evolves lookup tables that tell Robby what to do in each of the 243 situations he can see. The
random first strategies bump into walls and score far below zero. Within a few dozen generations the population learns
not to crash, and later it learns to pick up cans, so its best score keeps climbing in steps.
"""

# %%
# The population has 200 strategies, each scored over 10 sessions of 100 actions in a fresh room, and evolves for 290
# generations with a mutation rate of 0.005 per gene. The system logs only the size of the population, so after the run
# the mean and the best score of every generation are logged as time series, and the blueprint puts them first, as the
# primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.evolutionary_computation import RobbyGASystem

SYSTEM = 'robby_ga'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/scores', name='mean and best score', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
system = RobbyGASystem(n_strategies=200, mutation=0.005, random_state=0)
trajectory = simulate(system, until=290, dt=1, observers=[observer], random_state=0)
scores = np.array([trajectory.agents('strategies', frame).values['score'] for frame in range(len(trajectory.times))])
for name, series in (('mean', scores.mean(axis=1)), ('best', scores.max(axis=1))):
    observer.recording_.send_columns(
        f'scores/{name}',
        indexes=[rr.TimeColumn('sim_time', duration=trajectory.times)],
        columns=rr.Scalars.columns(scalars=series),
    )
observer.recording_.flush()

# %%
# Learning in stages
# ------------------
# A crash into a wall costs 5 points and an attempt to pick up a can where there is none costs 1, so random tables
# should score about -200. The mean score should rise to near 0 as the population learns to avoid walls, then the
# best score should climb above it in steps as strategies learn to collect cans, worth 10 points each.

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(trajectory.times, scores.mean(axis=1), label='mean score')
ax.plot(trajectory.times, scores.max(axis=1), label='best score')
ax.fill_between(trajectory.times, *np.percentile(scores, [25, 75], axis=1), alpha=0.3, label='middle half')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('generation')
ax.set_ylabel('score per session')
ax.set_title('Robby learns to avoid walls, then to collect cans')
ax.legend()
plt.show()

# %%
# Evolution by circumstance
# -------------------------
# Gooding replaces the selection of the genetic algorithm with deaths: strategies live seven generations, the worst 30%
# starve every generation, and random survivors, whatever their scores, breed in the places of the dead. Switching the
# starvation on and off every 30 generations should make the mean score climb while it is on and stall or fall back
# while it is off, and without any starvation the scores should stay near those of random tables.

fig, ax = plt.subplots(figsize=(7, 4.5))
for label, settings in (
    ('starvation switched every 30', {'starvation': 0.3, 'switch': 30}),
    ('no starvation', {'starvation': 0.0}),
):
    run = simulate(RobbyGASystem(regime='circumstance', **settings), until=120, dt=1)
    means = [run.agents('strategies', frame).values['score'].mean() for frame in range(121)]
    ax.plot(run.times, means, label=label)
for start in (30, 90):
    ax.axvspan(start, start + 30, color='grey', alpha=0.2)
ax.set_xlabel('generation (starvation off in the grey bands)')
ax.set_ylabel('mean score per session')
ax.set_title('Evolution by circumstance')
ax.legend()
plt.show()
