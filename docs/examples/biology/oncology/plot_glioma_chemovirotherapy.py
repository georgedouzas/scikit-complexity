"""
# Glioma under temozolomide and an oncolytic virus

Untreated, the glioma grows by more than an order of magnitude, crowds out the glia, and the neurons die with the glia.
"""

# %%
# The model is created by name from the catalog with the brain of Urenda-Cázares et al.: 5.94e12 glial cells and
# neurons and a glioma of 6e10 cells. This run gives neither drug nor virus. The observer saves every second frame
# with the default blueprint, whose primary view is the time series of the cells.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'glioma_chemovirotherapy'

system = create(MODEL, V=0.0, u1=0.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=300)

# %%
# Therapy spares the neurons
# --------------------------
# The constant infusion of 400 mg per m² a day, the virus, and both should each clear the glioma. The neurons should
# then stay above 98% of their initial number, while untreated they lose about a third.

arms = {
    'temozolomide': create(MODEL, V=0.0),
    'virus': create(MODEL, u1=0.0),
    'both': create(MODEL),
}
untreated = simulation.trajectory_
fig, (glioma, neurons) = plt.subplots(1, 2, figsize=(10, 4))
glioma.semilogy(untreated.times, untreated['brain.AU'], label='untreated', color='black')
neurons.plot(untreated.times, untreated['brain.N'] / untreated['brain.N'][0], label='untreated', color='black')
for name, arm in arms.items():
    trajectory = Simulation(arm, dt=1.0).run(until=300).trajectory_
    glioma.semilogy(trajectory.times, trajectory['brain.AU'].clip(1.0), label=name)
    neurons.plot(trajectory.times, trajectory['brain.N'] / trajectory['brain.N'][0], label=name)
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
