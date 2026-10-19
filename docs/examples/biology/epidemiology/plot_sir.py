"""
# SIR

A single outbreak: the infected grow, peak when the susceptibles fall below the threshold gamma / beta, and die out,
leaving part of the population never infected. Every infection and recovery is a random event of one individual, so
the curves are jagged.
"""

# %%
# The model runs as a jump process. The blueprint shows the three compartments in one time series view over the
# whole run, in the usual colours.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sir'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
colors = {'S': [70, 130, 230], 'I': [230, 60, 60], 'R': [80, 190, 90]}
compartments = rrb.TimeSeriesView(
    origin='/',
    contents=[f'/{name}/**' for name in colors],
    name='S, I, R',
    defaults=[rr.SeriesLines.from_fields(widths=2.5)],
    axis_x=rrb.TimeAxis(view_range=whole_run),
    overrides={f'{name}/n': rr.SeriesLines.from_fields(colors=color).visualizer() for name, color in colors.items()},
)
blueprint = rrb.Blueprint(
    compartments,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(create(MODEL, stochastic=True), dt=1.0, observers=[observer], random_state=0).run(until=160)

# %%
# Final size of the epidemic
# --------------------------
# For `dS/dt = -beta S I` and `dI/dt = beta S I - gamma I`, the susceptibles left at the end solve the final-size
# equation `S = S0 exp(-(beta / gamma) (N - S))`, with `N` the population. The simulated susceptibles should settle
# close to its solution, up to the randomness of a single outbreak.

trajectory = simulation.trajectory_
susceptible, infected = trajectory['S.n'], trajectory['I.n']
total = susceptible[0] + infected[0] + trajectory['R.n'][0]
ratio = 0.0003 / 0.1
final = brentq(lambda s: s - susceptible[0] * np.exp(-ratio * (total - s)), 1e-9, susceptible[0] - 1e-9)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trajectory.times, susceptible, label='susceptible', color='tab:blue')
ax.plot(trajectory.times, infected, label='infected', color='tab:red')
ax.axhline(final, color='black', linestyle='--', label=f'final-size equation: {final:.0f}')
ax.set_xlabel('time')
ax.set_ylabel('individuals')
ax.set_title(f'Susceptibles left: {susceptible[-1]:.0f} simulated, {final:.0f} predicted')
ax.legend()
plt.show()
