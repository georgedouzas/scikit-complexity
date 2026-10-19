"""
# Delayed metastasis

The primary tumour outcompetes the normal cells within a day, and the tumour cells that migrate after a delay of one
day seed a secondary tumour that does the same at the second site.
"""

# %%
# The model is created by name from the catalog, untreated, from 165 tumour cells at the primary site and none at the
# secondary site. The observer saves every frame with the default blueprint, whose primary view is the time series of
# the cells at both sites.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'delayed_metastasis'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=6)

# %%
# Takeover at both sites
# ----------------------
# Each tumour should settle at its logistic capacity, reduced at the primary site by the cells that leave,
# `(1 - delta / r2) / b2`, and set by `1 / b4` at the secondary site, while the normal cells die out.

trajectory = simulation.trajectory_
times = trajectory.times
fig, (primary, secondary) = plt.subplots(1, 2, figsize=(10, 4))
primary.plot(times, trajectory['patient.Np'], label='normal cells')
primary.plot(times, trajectory['patient.Ap'], label='tumour cells')
primary.axhline((1 - 0.1 / 10.0) / 0.00047, color='grey', linestyle=':', label='(1 - delta / r2) / b2')
primary.set_xlabel('time (days)')
primary.set_ylabel('cells')
primary.set_title('Primary site')
primary.legend()
secondary.plot(times, trajectory['patient.Ns'], label='normal cells')
secondary.plot(times, trajectory['patient.As'], label='tumour cells')
secondary.axhline(1 / 0.00043, color='grey', linestyle=':', label='1 / b4')
secondary.set_xlabel('time (days)')
secondary.set_ylabel('cells')
secondary.set_title('Secondary site, seeded after the delay')
secondary.legend()
fig.tight_layout()
plt.show()
