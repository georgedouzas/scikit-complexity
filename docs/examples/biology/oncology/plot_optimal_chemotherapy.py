"""
# Optimal chemotherapy

An oncologist doses a drug that kills tumour, immune, and normal cells, to minimise the tumour burden plus the toxicity
of the drug. The optimal infusion starts at the largest dose, holds it while the tumour is large, and tapers off, and
the tumour shrinks to well below its untreated size.
"""

# %%
# The model is created by name from the catalog with the de Pillis-Radunskaya cells of the book and the weight 0.1 of
# the squared infusion. The Pontryagin solver of its catalog entry solves the problem over 30 days. The default
# blueprint shows the cells, the drug, and the infusion rate in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.system import System
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe
from skcomplex.models.biology.oncology import TumourImmuneChemotherapy

MODEL = 'optimal_chemotherapy'

system = create(MODEL)
entry = describe(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=0.25,
    solver=entry.defaults['solver'],
    observers=[observer],
    observables=entry.observables_of(system),
).run(until=30)

# %%
# Treated against untreated
# -------------------------
# Without the drug the tumour keeps growing towards its capacity. With the optimal infusion, at its bound of 1 for the
# first days, the tumour should end at less than half its untreated size, while the normal cells recover once the
# infusion tapers off.

untreated = Simulation(System(components=[TumourImmuneChemotherapy('patient')]), dt=0.25).run(until=30).trajectory_
trajectory = simulation.trajectory_
fig, (tumour, dose) = plt.subplots(1, 2, figsize=(10, 4))
tumour.plot(untreated.times, untreated['patient.A'], label='untreated')
tumour.plot(trajectory.times, trajectory['patient.A'], label='optimal infusion')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells A')
tumour.set_title('The tumour shrinks under treatment')
tumour.legend()
dose.plot(trajectory.times, trajectory['patient.u'], label='infusion u')
dose.plot(trajectory.times, trajectory['patient.N'], label='normal cells N')
dose.set_xlabel('time (days)')
dose.set_title('Full dose first, then less')
dose.legend()
fig.tight_layout()
plt.show()
