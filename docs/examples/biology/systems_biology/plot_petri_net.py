"""
# Petri net

A Petri net of gene expression: tokens on places stand for the gene state, mRNA, and protein, and transitions consume
and produce them as they fire at random. The protein switches its own gene off, and the gene is on or off at every
moment, a place invariant of the net.
"""

# %%
# The system is created by name from the catalog with the gene expression net, and the observer saves every frame with
# the default blueprint, whose primary view is the time series of the tokens.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'petri_net'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer], random_state=0).prepare().run(until=100)

# %%
# Stochastic and continuous firing
# --------------------------------
# The stochastic net should fluctuate around the continuous Petri net of the same transitions, and the place invariant
# `gene_on + gene_off` should stay at 1.

continuous = Simulation(create_system(SYSTEM, stochastic=False), dt=0.5).prepare().run(until=100).trajectory_
trajectory = simulation.trajectory_
fig, (tokens, invariant) = plt.subplots(1, 2, figsize=(11, 4.5))
tokens.step(trajectory.times, trajectory.values('protein.n'), where='post', label='protein, stochastic')
tokens.plot(continuous.times, continuous.values('protein.n'), label='protein, continuous')
tokens.step(trajectory.times, trajectory.values('mRNA.n'), where='post', label='mRNA, stochastic')
tokens.set_xlabel('time')
tokens.set_ylabel('tokens')
tokens.set_title('Firing of the transitions')
tokens.legend()
invariant.step(trajectory.times, trajectory.values('gene_on.n') + trajectory.values('gene_off.n'), where='post')
invariant.step(trajectory.times, trajectory.values('gene_on.n'), where='post', alpha=0.6)
invariant.set_xlabel('time')
invariant.set_ylabel('gene tokens')
invariant.set_title('gene_on + gene_off stays 1')
fig.tight_layout()
plt.show()
