"""
# STDP network

Theta neurons on a random directed network change their synapses by the timing of their spikes: a synapse grows when
its presynaptic neuron fires just before its postsynaptic neuron and shrinks when it fires just after. Neurons that fire
at different rates meet at all delays, so the window's balance of potentiation and depression decides the drift.
"""

# %%
# The system is created by name from the catalog with a window dominated by depression. The observer saves every other
# frame with the default blueprint, whose primary view is the network with its synapses.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'stdp_network'

system = create_system(SYSTEM, A_minus=1.3)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, observers=[observer]).prepare().run(until=150)

# %%
# Drift of the efficacies
# -----------------------
# The mean efficacy should fall when depression dominates the window, `A_minus > A_plus`, and rise when potentiation
# dominates, while the efficacies spread out.

fig, (drift, spread) = plt.subplots(1, 2, figsize=(11, 4.5))
for a_minus in (0.6, 1.0, 1.3):
    run = Simulation(create_system(SYSTEM, A_minus=a_minus), dt=0.5).prepare().run(until=150).trajectory_
    drift.plot(run.times, run.values('synapses.kappa').mean(axis=1), label=f'A_minus = {a_minus}')
drift.set_xlabel('time')
drift.set_ylabel('mean efficacy')
drift.set_title('The window decides the drift')
drift.legend()
final = simulation.trajectory_.values('synapses.kappa')[-1]
spread.hist(final, bins=20)
spread.set_xlabel('efficacy at the end')
spread.set_ylabel('synapses')
spread.set_title('Efficacies of the run above')
fig.tight_layout()
plt.show()
