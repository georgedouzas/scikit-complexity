"""
# Linear-quadratic fractions with a radiosensitizer

Ten daily fractions of 2 Gy each move a fixed share of the proliferating cells into the damaged compartments, where
they divide once more and die over the following weeks, while the survivors regrow more slowly.
"""

# %%
# The model is created by name from the catalog: a xenograft of 27 mm³ receives ten fractions of 2 Gy, the first
# three hours after a radiosensitizer at 2 μg/mL. The observer saves every frame with the default blueprint, whose
# primary view is the time series of the six compartments over the 40 days of the response.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'lq_radiosensitizer'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.25, observers=[observer]).run(until=40)

# %%
# Surviving fractions
# -------------------
# A fraction of dose `d` leaves `exp(-(1 + ap Us)(alpha d + beta d^2))` of the proliferating cells, so without the
# radiosensitizer every fraction leaves `exp(-alpha d - beta d^2) = 0.878`. The radiosensitizer lowers the survival of
# the first fractions, while it lasts, and the treated tumour stays smaller than the one irradiated alone.

alpha, beta, ap, dose = 0.054, 0.0054, 0.42, 2.0
trajectory = simulation.trajectory_
names = list(trajectory.problem.states.names)
cells, sensitizer = names.index('tumour.A1'), names.index('tumour.Us')
survival = [event.after[cells] / event.before[cells] for event in trajectory.events]
predicted = [
    np.exp(-(1 + ap * event.before[sensitizer]) * (alpha * dose + beta * dose**2)) for event in trajectory.events
]
alone = Simulation(create(MODEL, Us=0.0), dt=0.25).run(until=40).trajectory_


def total(run):
    """Return the total tumour volume of a run."""
    return sum(run[f'tumour.A{index}'] for index in range(1, 7))


fig, (bars, volume) = plt.subplots(1, 2, figsize=(10, 4))
bars.bar(np.arange(1, len(survival) + 1), survival, label='simulated')
bars.plot(np.arange(1, len(predicted) + 1), predicted, 'o', color='black', label='linear-quadratic')
bars.axhline(np.exp(-alpha * dose - beta * dose**2), color='grey', linestyle=':', label='without radiosensitizer')
bars.set_ylim(0.7, 0.9)
bars.set_xlabel('fraction')
bars.set_ylabel('surviving fraction of A1')
bars.set_title('Each fraction leaves the linear-quadratic share')
bars.legend()
volume.plot(trajectory.times, total(trajectory), label='with radiosensitizer')
volume.plot(alone.times, total(alone), label='radiation alone')
volume.set_xlabel('time (days)')
volume.set_ylabel('tumour volume (mm³)')
volume.set_title('The tumour shrinks, then regrows')
volume.legend()
fig.tight_layout()
plt.show()
