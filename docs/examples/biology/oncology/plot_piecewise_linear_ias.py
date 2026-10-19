"""
# Piecewise-linear intermittent androgen suppression

The PSA of a patient of type 1 falls during therapy and rises after it, so the therapy cycles between 10 and 4 ng/mL
without relapse.
"""

# %%
# The model is created by name from the catalog for patient type 1, whose PSA of 19 ng/mL starts the therapy at once.
# The observer saves every second frame with the default blueprint, whose primary view is the time series of the three
# cell populations and the therapy.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'piecewise_linear_ias'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=5.0, observers=[observer]).run(until=1500)

# %%
# Three patient types
# -------------------
# The three types differ in how fast their cells mutate during therapy. Type 1 should keep cycling, type 2 should
# cycle with off periods that grow shorter, and type 3 should relapse, its PSA rising under therapy once the mutated
# cells outgrow it.

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for patient, axis in zip((1, 2, 3), axes, strict=True):
    run = simulation if patient == 1 else Simulation(create(MODEL, patient=patient), dt=5.0).run(until=1500)
    trajectory = run.trajectory_
    psa = trajectory['tumour.AD'] + trajectory['tumour.AIR'] + trajectory['tumour.AII']
    axis.plot(trajectory.times, psa)
    axis.fill_between(trajectory.times, 0, 25, where=trajectory['tumour.u'] > 0.5, color='red', alpha=0.15)
    axis.axhline(10.0, color='black', linestyle='--')
    axis.axhline(4.0, color='grey', linestyle='--')
    axis.set_xlabel('time (days)')
    axis.set_title(f'Patient type {patient}')
axes[0].set_ylabel('PSA (ng/mL), therapy shaded')
fig.tight_layout()
plt.show()
