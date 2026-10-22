"""
# Bevacizumab and atezolizumab against a vascularized tumour

Every three weeks both drugs are infused, the vasculature collapses step by step, the effector cells rise, and the
tumour shrinks to about a third of its initial volume.
"""

# %%
# The model is created by name from the catalog with the renal cell carcinoma of Teles and Lemos: a tumour of
# 5000 mm³ on a vasculature of 625 mm³, under 15 mg per kg of bevacizumab and 20 mg per kg of atezolizumab every
# 21 days. The observer saves every second frame with the default blueprint, whose primary view is the time series of
# the tumour, the vasculature, the effector cells, and the drugs.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'angiogenesis_immunotherapy'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, observers=[observer], observables=describe(MODEL).observables_of(system)).run(
    until=150,
)

# %%
# Each drug alone and both together
# ---------------------------------
# Untreated, the tumour grows toward its vascular capacity. Bevacizumab alone should hold it near its initial volume
# and atezolizumab alone should slow it, while the combination should shrink it. The toxicity peaks at every infusion
# and the drug amounts fall between doses.

arms = {
    'untreated': create(MODEL, dose_an=0.0, dose_im=0.0),
    'bevacizumab alone': create(MODEL, dose_im=0.0),
    'atezolizumab alone': create(MODEL, dose_an=0.0),
}
fig, (tumour, toxicity) = plt.subplots(1, 2, figsize=(10, 4))
for name, arm in arms.items():
    trajectory = Simulation(arm, dt=0.5, observables=describe(MODEL).observables_of(arm)).run(until=150).trajectory_
    tumour.plot(trajectory.times, trajectory['tumour.A'], label=name)
combined = simulation.trajectory_
tumour.plot(combined.times, combined['tumour.A'], label='combination', color='black')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour volume (mm³)')
tumour.set_title('The combination shrinks the tumour')
tumour.legend()
toxicity.plot(combined.times, combined['toxicity'], color='black', label='mean toxicity')
toxicity.plot(combined.times, combined['tumour.c_an'] / 15, label='bevacizumab / dose')
toxicity.plot(combined.times, combined['tumour.c_im'] / 20, label='atezolizumab / dose')
toxicity.set_xlabel('time (days)')
toxicity.set_ylabel('toxicity and relative amount')
toxicity.set_title('Doses every three weeks')
toxicity.legend()
fig.tight_layout()
plt.show()
