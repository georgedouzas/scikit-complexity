"""
# Petri net

A Petri net of gene expression: tokens on places stand for the gene state, mRNA, and protein, and transitions consume
and produce them as they fire at random. The protein switches its own gene off, and the gene is on or off at every
moment, a place invariant of the net.
"""

# %%
# The system is built from its class with the gene expression net, and the observer saves every frame with the default
# blueprint, whose primary view is the time series of the tokens.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import PetriNetSystem

SYSTEM = 'petri_net'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(PetriNetSystem(), observers=[observer], random_state=0)

# %%
# Stochastic and continuous firing
# --------------------------------
# The stochastic net should fluctuate around the continuous Petri net of the same transitions, and the place invariant
# `gene_on + gene_off` should stay at 1.

continuous = simulate(PetriNetSystem(stochastic=False))
fig, (tokens, invariant) = plt.subplots(1, 2, figsize=(11, 4.5))
tokens.step(trajectory.times, trajectory.get_values('protein.x'), where='post', label='protein, stochastic')
tokens.plot(continuous.times, continuous.get_values('protein.x'), label='protein, continuous')
tokens.step(trajectory.times, trajectory.get_values('mRNA.x'), where='post', label='mRNA, stochastic')
tokens.set_xlabel('time')
tokens.set_ylabel('tokens')
tokens.set_title('Firing of the transitions')
tokens.legend()
invariant.step(trajectory.times, trajectory.get_values('gene_on.x') + trajectory.get_values('gene_off.x'), where='post')
invariant.step(trajectory.times, trajectory.get_values('gene_on.x'), where='post', alpha=0.6)
invariant.set_xlabel('time')
invariant.set_ylabel('gene tokens')
invariant.set_title('gene_on + gene_off stays 1')
fig.tight_layout()
plt.show()
