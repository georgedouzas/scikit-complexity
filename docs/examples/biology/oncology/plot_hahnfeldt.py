"""
# Hahnfeldt angiogenesis system

A tumour at the equilibrium of its vasculature regresses under a continuous infusion of endostatin, which prunes the
vasculature that sets its carrying capacity.
"""

# %%
# The system is built from its class with its defaults: tumour and vascular volumes at the equilibrium of 17347 mm³
# and a continuous infusion of 20 mg/kg/day of endostatin from day 0. The observer saves every frame with the default
# blueprint, whose primary view is the time series of the volumes and the inhibitor.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import HahnfeldtSystem

SYSTEM = 'hahnfeldt'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(HahnfeldtSystem(), dt=0.2, observers=[observer])

# %%
# Equilibrium and regression
# --------------------------
# Untreated, a small tumour and its vasculature should both approach `(xi_s / xi_i)^(3/2)`, about 17347 mm³. Under the
# infusion, the inhibitor should settle at `dose / du`, and the vasculature and then the tumour should shrink.

untreated = simulate(HahnfeldtSystem(A=200.0, Vb=625.0, dose=0.0), dt=1.0, until=100)
treated = trajectory
equilibrium = (5.85 / 0.00873) ** 1.5
fig, (growth, therapy) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(untreated.times, untreated.get_values('tumour.A'), label='tumour')
growth.plot(untreated.times, untreated.get_values('tumour.Vb'), label='vasculature')
growth.axhline(equilibrium, color='black', linestyle='--', label='(xi_s / xi_i)^(3/2)')
growth.set_xlabel('time (days)')
growth.set_ylabel('volume (mm³)')
growth.set_title('Untreated, both approach the equilibrium')
growth.legend()
therapy.semilogy(treated.times, treated.get_values('tumour.A'), label='tumour')
therapy.semilogy(treated.times, treated.get_values('tumour.Vb'), label='vasculature')
therapy.semilogy(treated.times, np.full_like(treated.times, equilibrium), 'k--', label='untreated equilibrium')
therapy.set_xlabel('time (days)')
therapy.set_ylabel('volume (mm³)')
therapy.set_title('Endostatin makes both regress')
therapy.legend()
fig.tight_layout()
plt.show()
