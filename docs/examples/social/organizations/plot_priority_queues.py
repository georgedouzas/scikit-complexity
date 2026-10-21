"""
# Barabási priority queues

A person keeps a list of two tasks and almost always executes the one of highest priority. Most tasks are done at
once, but a task of low priority can wait for thousands of steps, and the waiting times follow a power law with an
exponent near 1, as Barabási found for the timing of e-mails and letters.
"""

# %%
# The model is created by name from the catalog with two tasks, and the highest priority task is executed with
# probability 0.999 and a random task otherwise, so that waits are cut off near 1 / (1 - 0.999) = 1000 steps. It runs
# for 100000 steps, one task executed per step. The default blueprint shows the age of both tasks, sampled every 400
# steps: one task is executed at every step while the other ages along long ramps until a new task of even lower
# priority frees it.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fit_power_law
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'priority_queues'

system = create(MODEL, greed=0.999, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=400)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=100_000)

# %%
# Distribution of waiting times
# -----------------------------
# The execution reports the waiting time of the task it executes at every step as `execution.wait`. Barabási's model
# gives a power law of exponent 1 for waits well below the cutoff of 1000 steps, so the fitted exponent of the waits
# from 2 to 300 steps should lie near 1, and the log-binned distribution should fall by about one decade per decade
# before it bends down past the cutoff.

waits = simulation.trajectory_['execution.wait'][1:]
fit = fit_power_law(waits, discrete=True, xmin=2, xmax=300, bootstrap=0)
edges = np.unique(np.logspace(0, np.log10(waits.max() + 1), 20).astype(int))
counts, _ = np.histogram(waits, edges)
density = counts / np.diff(edges) / len(waits)
centres = np.sqrt(edges[:-1] * edges[1:])
shown = density > 0
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(centres[shown], density[shown], 'o', label=f'{len(waits)} executed tasks')
guide = density[1] * (centres[1] / centres[1:]) ** fit.alpha
ax.loglog(centres[1:], guide, '--', color='gray', label=f'power law, fitted exponent {fit.alpha:.2f}')
ax.axvline(1000, color='C3', linewidth=1, label='cutoff 1 / (1 - greed)')
ax.set_xlabel('waiting time (steps)')
ax.set_ylabel('probability')
ax.set_title('Priority queues give heavy-tailed waiting times')
ax.legend()
plt.show()
