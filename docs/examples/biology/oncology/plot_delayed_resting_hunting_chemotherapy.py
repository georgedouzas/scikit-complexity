"""
# Delayed immune recruitment under chemotherapy

The delay with which resting immune cells become hunting cells makes the tumour rise and collapse in irregular bursts,
and a constant drug infusion ends the bursts by clearing the tumour in one of its troughs.
"""

# %%
# The model is created by name from the catalog with a constant infusion of 0.03, the top of the book's range, and the
# conversion delay of 45.6. The observer saves every frame with the default blueprint, whose primary view is the time
# series of the cells and the drug.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'delayed_resting_hunting_chemotherapy'

system = create(MODEL, u=0.03)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=5.0, observers=[observer]).run(until=1000)

# %%
# Oscillations and clearance
# --------------------------
# Untreated, the tumour and the hunting cells should keep oscillating out of phase, with the tumour repeatedly nearly
# cleared and regrowing. With the infusion, the drug should finish the tumour off in its first deep trough.

untreated = Simulation(create(MODEL), dt=5.0).run(until=1000).trajectory_
treated = simulation.trajectory_
fig, (tumour, phase) = plt.subplots(1, 2, figsize=(10, 4))
tumour.plot(untreated.times, untreated['patient.A'], label='untreated')
tumour.plot(treated.times, treated['patient.A'], label='infusion u = 0.03')
tumour.set_xlabel('time')
tumour.set_ylabel('tumour cells')
tumour.set_title('The infusion ends the bursts')
tumour.legend()
phase.plot(untreated['patient.A'], untreated['patient.E'])
phase.set_xlabel('tumour cells')
phase.set_ylabel('hunting cells')
phase.set_title('Untreated, the cells oscillate')
fig.tight_layout()
plt.show()
