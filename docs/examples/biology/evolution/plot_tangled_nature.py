"""
# Tangled Nature

Organisms of many genome types help or harm each other through random couplings. Mutants explore the genome space,
and the ecosystem settles into quasi-stable configurations of a few dominant types that last for a while and then
collapse abruptly into new ones, a punctuated equilibrium.
"""

# %%
# One hundred organisms of a single random type start the run, with genomes of eight bits. The model's blueprint shows
# the number of distinct genome types alive, which jumps from one to about forty as mutants spread and then fluctuates
# around thirty. The abrupt transitions between quasi-stable configurations take thousands of generations, longer than
# this run.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tangled_nature'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=300)

# %%
# Population and dominant types
# -----------------------------
# The population should grow from 100 until the couplings and the limit of resources hold it near 340, and the
# single type of the start should give way to a coexisting set in which the most common type holds about a third.

trajectory = simulation.trajectory_
frames = range(len(trajectory.times))
genomes = [trajectory.agents('organisms', frame).values['trait'] for frame in frames]
sizes = np.array([len(genome) for genome in genomes])
dominant = np.array([np.unique(genome, return_counts=True)[1].max() / len(genome) for genome in genomes])
fig, (population, share) = plt.subplots(1, 2, figsize=(11, 4))
population.plot(trajectory.times, sizes)
population.set_xlabel('step')
population.set_ylabel('organisms')
population.set_title('The population levels off')
share.plot(trajectory.times, dominant)
share.set_ylim(0, 1)
share.set_xlabel('step')
share.set_ylabel('share of the most common type')
share.set_title('A coexisting set of types replaces the founder')
fig.tight_layout()
plt.show()
