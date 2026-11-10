"""
# Piecewise-linear intermittent androgen suppression

The PSA of a patient of type 1 falls during therapy and rises after it, so the therapy cycles between 10 and 4 ng/mL
without relapse.
"""

# %%
# The example builds `PiecewiseLinearIASSystem` for patient type 1, whose PSA of 19 ng/mL starts the therapy at once.
# The observer saves every second frame with the default blueprint, whose primary view is the time series of the three
# cell populations and the therapy.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import PiecewiseLinearIASSystem

SYSTEM = 'piecewise_linear_ias'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(PiecewiseLinearIASSystem(), observers=[observer])

# %%
# Three patient types
# -------------------
# The three types differ in how fast their cells mutate during therapy. Type 1 should keep cycling, type 2 should
# cycle with off periods that grow shorter, and type 3 should relapse, its PSA rising under therapy once the mutated
# cells outgrow it.

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for patient, axis in zip((1, 2, 3), axes, strict=True):
    run = trajectory if patient == 1 else simulate(PiecewiseLinearIASSystem(patient=patient))
    psa = run.get_values('tumour.AD') + run.get_values('tumour.AIR') + run.get_values('tumour.AII')
    axis.plot(run.times, psa)
    axis.fill_between(run.times, 0, 25, where=run.get_values('tumour.u') > 0.5, color='red', alpha=0.15)
    axis.axhline(10.0, color='black', linestyle='--')
    axis.axhline(4.0, color='grey', linestyle='--')
    axis.set_xlabel('time (days)')
    axis.set_title(f'Patient type {patient}')
axes[0].set_ylabel('PSA (ng/mL), therapy shaded')
fig.tight_layout()
plt.show()
