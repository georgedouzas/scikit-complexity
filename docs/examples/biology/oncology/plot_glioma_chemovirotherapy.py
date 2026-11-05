"""
# Glioma under temozolomide and an oncolytic virus

Untreated, the glioma grows by more than an order of magnitude, crowds out the glia, and the neurons die with the glia.
"""

# %%
# The system is created by name from the catalog with the brain of Urenda-Cázares et al.: 5.94e12 glial cells and
# neurons and a glioma of 6e10 cells. This run gives neither drug nor virus. The observer saves every second frame
# with the default blueprint, whose primary view is the time series of the cells.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'glioma_chemovirotherapy'

system = create_system(SYSTEM, V=0.0, u1=0.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).prepare().run(until=300)

# %%
# Therapy spares the neurons
# --------------------------
# The constant infusion of 400 mg per m² a day, the virus, and both should each clear the glioma. The neurons should
# then stay above 98% of their initial number, while untreated they lose about a third.

arms = {
    'temozolomide': create_system(SYSTEM, V=0.0),
    'virus': create_system(SYSTEM, u1=0.0),
    'both': create_system(SYSTEM),
}
untreated = simulation.trajectory_
fig, (glioma, neurons) = plt.subplots(1, 2, figsize=(10, 4))
glioma.semilogy(untreated.times, untreated.values('brain.AU'), label='untreated', color='black')
neurons.plot(
    untreated.times,
    untreated.values('brain.N') / untreated.values('brain.N')[0],
    label='untreated',
    color='black',
)
for name, arm in arms.items():
    trajectory = Simulation(arm, dt=1.0).prepare().run(until=300).trajectory_
    glioma.semilogy(trajectory.times, trajectory.values('brain.AU').clip(1.0), label=name)
    neurons.plot(trajectory.times, trajectory.values('brain.N') / trajectory.values('brain.N')[0], label=name)
glioma.set_xlabel('time (days)')
glioma.set_ylabel('glioma cells')
glioma.set_title('Each therapy clears the glioma')
glioma.legend()
neurons.axhline(0.98, color='grey', linestyle=':', label='98% of the neurons')
neurons.set_xlabel('time (days)')
neurons.set_ylabel('neurons (fraction of initial)')
neurons.set_title('Therapy spares the neurons')
neurons.legend()
fig.tight_layout()
plt.show()
