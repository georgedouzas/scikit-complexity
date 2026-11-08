"""
# Mark-0 economy and a Covid shock

Firms hire, fire, and set prices and wages by rules of thumb, a household consumes a share of its savings and income,
and a bank lends to the firms. A lockdown cuts the household's consumption, and when it lasts long enough the debts of
the firms push them into bankruptcy, which tips the economy into a lasting recession unless credit stays easy.
"""

# %%
# Five hundred firms run for 300 months from the baseline of Table 1. From month 100 the consumption propensity falls
# by half for nine months. The default blueprint shows the reporters of the economy without policy.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.macroeconomics import Mark0System

SYSTEM = 'mark0'
SHOCK = {'shock_length': 9, 'dc': 0.5}

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulate(Mark0System(**SHOCK), observers=[observer])

# %%
# Recovery shapes under the policies
# ----------------------------------
# A mild shock should end in a quick V-shaped recovery. A strong one should leave unemployment high for good, the
# L shape, while lifting the bankruptcy threshold during the shock and lowering it with the mean fragility afterwards,
# the adaptive policy of Eq. 24, should bring the economy back to full employment.

scenarios = {
    'no shock': {},
    'mild shock': {'shock_length': 6, 'dc': 0.3},
    'strong shock': SHOCK,
    'strong shock, adaptive credit': {**SHOCK, 'credit': 'adaptive'},
}
fig, (unemployment, bankruptcies) = plt.subplots(1, 2, figsize=(10, 4))
for name, settings in scenarios.items():
    trajectory = simulate(Mark0System(**settings))
    unemployment.plot(trajectory.times, trajectory.values('economy.unemployment'), label=name)
    bankruptcies.plot(trajectory.times, trajectory.values('economy.bankruptcies'), label=name)
for axis, label in ((unemployment, 'unemployment rate'), (bankruptcies, 'bankruptcies per month')):
    axis.axvspan(100, 109, color='grey', alpha=0.2)
    axis.set_xlabel('month')
    axis.set_ylabel(label)
unemployment.set_title('Recovery shapes')
unemployment.legend()
bankruptcies.set_title('Bankruptcies')
fig.tight_layout()
plt.show()
