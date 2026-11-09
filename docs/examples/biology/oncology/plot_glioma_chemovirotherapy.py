"""
# Glioma under temozolomide and an oncolytic virus

Untreated, the glioma grows by more than an order of magnitude, crowds out the glia, and the neurons die with the glia.
"""

# %%
# The system is built from its class with the brain of Urenda-Cázares et al.: 5.94e12 glial cells and neurons and a
# glioma of 6e10 cells. This run gives neither drug nor virus. The observer saves every second frame with the default
# blueprint, whose primary view is the time series of the cells.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import GliomaChemovirotherapySystem

SYSTEM = 'glioma_chemovirotherapy'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(GliomaChemovirotherapySystem(V=0.0, u1=0.0), observers=[observer])

# %%
# Therapy spares the neurons
# --------------------------
# The constant infusion of 400 mg per m² a day, the virus, and both should each clear the glioma. The neurons should
# then stay above 98% of their initial number, while untreated they lose about a third.

arms = {
    'temozolomide': GliomaChemovirotherapySystem(V=0.0).prepare().system_,
    'virus': GliomaChemovirotherapySystem(u1=0.0).prepare().system_,
    'both': GliomaChemovirotherapySystem().prepare().system_,
}
untreated = trajectory
fig, (glioma, neurons) = plt.subplots(1, 2, figsize=(10, 4))
glioma.semilogy(untreated.times, untreated.get_values('brain.AU'), label='untreated', color='black')
neurons.plot(
    untreated.times,
    untreated.get_values('brain.N') / untreated.get_values('brain.N')[0],
    label='untreated',
    color='black',
)
for name, arm in arms.items():
    trajectory = simulate(arm, dt=1.0, until=300)
    glioma.semilogy(trajectory.times, trajectory.get_values('brain.AU').clip(1.0), label=name)
    neurons.plot(trajectory.times, trajectory.get_values('brain.N') / trajectory.get_values('brain.N')[0], label=name)
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
