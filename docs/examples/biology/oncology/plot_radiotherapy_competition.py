"""
# Normal and tumour competition under radiation

Periodic irradiation holds the tumour cells below their capacity, and the normal cells, which the untreated tumour
would push out, decline towards a level at which the two coexist.
"""

# %%
# The model is created by name from the catalog: normal cells at 0.5 and tumour cells at 0.8 of their capacities
# compete, and the tumour is irradiated five days a week at the per capita kill rate 0.2. Frames are a week apart, at
# the same phase of the schedule, and the observer saves every frame with the default blueprint, whose primary view is
# the time series of the two concentrations.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'radiotherapy_competition'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=7.0, observers=[observer]).run(until=1400)

# %%
# Coexistence instead of exclusion
# --------------------------------
# Averaged over a week, the radiation lowers the growth rate of the tumour from `r2` to `r2 - a2 width / period`. The
# coexistence equilibrium of the averaged competition solves `r1 (1 - N) = c1 A + epsilon` and
# `r2 (1 - A) = c2 N + a2 width / period`. The treated normal cells should approach it, while without radiation they
# should decay towards 0 as the tumour fills its capacity.

r1, r2, c1, c2, a2, epsilon = 0.008, 0.4, 0.01, 0.15, 0.2, 0.001
kill = a2 * 5 / 7
normal = (r1 - epsilon - c1 * (1 - kill / r2)) / (r1 - c1 * c2 / r2)
untreated = Simulation(create(MODEL, a2=0.0, epsilon=0.0), dt=7.0).run(until=1400).trajectory_
treated = simulation.trajectory_
fig, (cells, tumour) = plt.subplots(1, 2, figsize=(10, 4))
cells.plot(treated.times, treated['tissue.N'], label='irradiated')
cells.plot(untreated.times, untreated['tissue.N'], label='untreated')
cells.axhline(normal, color='black', linestyle='--', label='averaged coexistence')
cells.set_xlabel('time (days)')
cells.set_ylabel('normal cells N')
cells.set_title('Radiation lets the normal cells persist')
cells.legend()
tumour.plot(treated.times, treated['tissue.A'], label='irradiated')
tumour.plot(untreated.times, untreated['tissue.A'], label='untreated')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells A')
tumour.set_title('Radiation holds the tumour below its capacity')
tumour.legend()
fig.tight_layout()
plt.show()
