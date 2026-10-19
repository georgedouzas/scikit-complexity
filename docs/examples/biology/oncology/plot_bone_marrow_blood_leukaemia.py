"""
# Leukaemia in the bone marrow and the blood

The normal marrow cells first grow toward their capacity, but the blasts that mutation and cancer stem cells feed
overtake them in the bone marrow, spill into the blood, and end up outnumbering them a hundredfold.
"""

# %%
# The model is created by name from the catalog with the initial cells of the book and rates within its ranges, and
# runs for 3000 hours with the maturation delay of 96 hours and the conversion delay of 576 hours. The observer saves
# every second frame with the default blueprint, whose primary view is the time series of the five populations.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bone_marrow_blood_leukaemia'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=10.0, solver=ScipyODE(method='BDF'), observers=[observer]).run(until=3000)

# %%
# Blasts crowd out the normal cells
# ---------------------------------
# On a logarithmic scale the five populations span eight orders of magnitude. The marrow blasts should overtake the
# normal marrow cells within two weeks, and the normal blood cells should fall by more than three orders of magnitude.

trajectory = simulation.trajectory_
days = trajectory.times / 24
fig, (marrow, blood) = plt.subplots(1, 2, figsize=(10, 4))
marrow.semilogy(days, trajectory['patient.NBM'], label='normal')
marrow.semilogy(days, trajectory['patient.ABM'], label='blasts')
marrow.semilogy(days, trajectory['patient.AS'], label='cancer stem cells')
marrow.set_xlabel('time (days)')
marrow.set_ylabel('cells')
marrow.set_title('Bone marrow')
marrow.legend()
blood.semilogy(days, trajectory['patient.NPB'], label='normal')
blood.semilogy(days, trajectory['patient.APB'], label='blasts')
blood.set_xlabel('time (days)')
blood.set_ylabel('cells')
blood.set_title('Peripheral blood')
blood.legend()
fig.tight_layout()
plt.show()
