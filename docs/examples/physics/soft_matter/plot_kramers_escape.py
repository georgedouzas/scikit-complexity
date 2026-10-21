"""
# Kramers escape

Particles start at the bottom of the left well of `U = x^4 / 4 - x^2 / 2` and wait there, jiggled by thermal noise,
until a rare large fluctuation carries them over the barrier of height `1 / 4` into the right well. The escapes are
independent rare events, so the left well empties exponentially at the rate Kramers computed. The view shows the
positions of six particles that rattle around `-1` or `+1` for long stretches and jump between the wells at random
times.
"""

# %%
# The model is created by name from the catalog with six particles and noise intensity `D = 0.07`. The Euler-Maruyama
# stepper takes steps of 0.01 and the observer saves every frame of 1.5 with the default blueprint, whose primary view
# is the time series of the positions.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'kramers_escape'

system = create(MODEL, n_particles=6, D=0.07, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.01)
simulation = Simulation(system, solver=solver, dt=1.5, observers=[observer], random_state=0).run(until=300)

# %%
# Exponential survival at the Kramers rate
# ----------------------------------------
# A particle has escaped once it first reaches the bottom of the right well at `x = 1`. For a low noise the share of
# particles that have not yet escaped should decay as `exp(-r t)`, a straight line on a semi-log plot. Kramers' rate
# `r = sqrt(U''(-1) |U''(0)|) / (2 pi) exp(-1 / (4 D))` is exact only for barriers much higher than `D`; at barriers
# of 2.5 to 3 times `D`, as for these 2000 particles, the measured slopes should fall about 15% below it.

fig, ax = plt.subplots(figsize=(7, 4))
for intensity, color in ((0.08, 'tab:blue'), (0.1, 'tab:orange')):
    run = Simulation(create(MODEL, n_particles=2000, D=intensity), solver=solver, dt=1.0, random_state=1).run(400)
    times = run.trajectory_.times
    escaped = np.maximum.accumulate(run.trajectory_['particles.x'] >= 1.0, axis=0)
    rate = np.sqrt(2) / (2 * np.pi) * np.exp(-1 / (4 * intensity))
    ax.semilogy(times, 1 - escaped.mean(axis=1), color=color, label=f'simulation, D = {intensity}')
    ax.semilogy(times, np.exp(-rate * times), '--', color=color, label=f'Kramers, r = {rate:.4f}')
ax.set_xlabel('time t')
ax.set_ylabel('share of particles not yet escaped')
ax.set_title('The left well empties at the Kramers rate')
ax.legend()
fig.tight_layout()
plt.show()
