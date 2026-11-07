"""
# Bevacizumab and atezolizumab against a vascularized tumour

Every three weeks both drugs are infused, the vasculature collapses step by step, the effector cells rise, and the
tumour shrinks to about a third of its initial volume.
"""

# %%
# The system is built from its class with the renal cell carcinoma of Teles and Lemos: a tumour of 5000 mm³ on a
# vasculature of 625 mm³, under 15 mg per kg of bevacizumab and 20 mg per kg of atezolizumab every 21 days. The
# observer saves every second frame with the default blueprint, whose primary view is the time series of the tumour,
# the vasculature, the effector cells, and the drugs.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import AngiogenesisImmunotherapySystem

SYSTEM = 'angiogenesis_immunotherapy'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
combined = simulate(AngiogenesisImmunotherapySystem(), dt=0.5, observers=[observer])

# %%
# Each drug alone and both together
# ---------------------------------
# Untreated, the tumour grows toward its vascular capacity. Bevacizumab alone should hold it near its initial volume
# and atezolizumab alone should slow it, while the combination should shrink it. The toxicity peaks at every infusion
# and the drug amounts fall between doses.

arms = {
    'untreated': AngiogenesisImmunotherapySystem(dose_an=0.0, dose_im=0.0),
    'bevacizumab alone': AngiogenesisImmunotherapySystem(dose_im=0.0),
    'atezolizumab alone': AngiogenesisImmunotherapySystem(dose_an=0.0),
}
fig, (tumour, toxicity) = plt.subplots(1, 2, figsize=(10, 4))
for name, arm in arms.items():
    trajectory = simulate(arm, dt=0.5)
    tumour.plot(trajectory.times, trajectory.values('tumour.A'), label=name)
tumour.plot(combined.times, combined.values('tumour.A'), label='combination', color='black')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour volume (mm³)')
tumour.set_title('The combination shrinks the tumour')
tumour.legend()
toxicity.plot(combined.times, combined.values('toxicity'), color='black', label='mean toxicity')
toxicity.plot(combined.times, combined.values('tumour.c_an') / 15, label='bevacizumab / dose')
toxicity.plot(combined.times, combined.values('tumour.c_im') / 20, label='atezolizumab / dose')
toxicity.set_xlabel('time (days)')
toxicity.set_ylabel('toxicity and relative amount')
toxicity.set_title('Doses every three weeks')
toxicity.legend()
fig.tight_layout()
plt.show()
