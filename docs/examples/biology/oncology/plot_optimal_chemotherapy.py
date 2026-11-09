"""
# Optimal chemotherapy

An oncologist doses a drug that kills tumour, immune, and normal cells, to minimise the tumour burden plus the toxicity
of the drug. The optimal infusion starts at the largest dose, holds it while the tumour is large, and tapers off, and
the tumour shrinks to well below its untreated size.
"""

# %%
# The system is built from its class with the de Pillis-Radunskaya cells of the book and the weight 0.1 of the squared
# infusion. Its tuned Pontryagin solver solves the problem over 30 days. The default blueprint shows
# the cells, the drug, and the infusion rate in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import OptimalChemotherapySystem, TumourImmuneChemotherapyElement

SYSTEM = 'optimal_chemotherapy'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(OptimalChemotherapySystem(), observers=[observer])

# %%
# Treated against untreated
# -------------------------
# Without the drug the tumour keeps growing towards its capacity. With the optimal infusion, at its bound of 1 for the
# first days, the tumour should end at less than half its untreated size, while the normal cells recover once the
# infusion tapers off.

untreated = simulate(CompositeSystem(elements=[TumourImmuneChemotherapyElement('patient')]), dt=0.25, until=30)
fig, (tumour, dose) = plt.subplots(1, 2, figsize=(10, 4))
tumour.plot(untreated.times, untreated.values('patient.A'), label='untreated')
tumour.plot(trajectory.times, trajectory.values('patient.A'), label='optimal infusion')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells A')
tumour.set_title('The tumour shrinks under treatment')
tumour.legend()
dose.plot(trajectory.times, trajectory.values('patient.u'), label='infusion u')
dose.plot(trajectory.times, trajectory.values('patient.N'), label='normal cells N')
dose.set_xlabel('time (days)')
dose.set_title('Full dose first, then less')
dose.legend()
fig.tight_layout()
plt.show()
