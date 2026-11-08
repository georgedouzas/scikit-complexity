"""
# Ajani's Choice

Chesney's model of labour exploitation in the greenhouses of Almería. Employers offer a daily wage, lower it when they
fill their workforce and raise it otherwise, and keep it in line with the wages of their neighbours. Migrant workers
refuse offers below the minimum wage unless they are desperate. When workers outnumber the jobs, underpayment emerges
without any employer planning it.
"""

# %%
# Ten employers need between 50 and 99 workers each, two slums hold 500 workers each, and the run lasts 60 days, two
# summer months. The default blueprint shows the wages of the employers and the pay of the workers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.organizations import AjanisChoiceSystem

SYSTEM = 'ajanis_choice'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(AjanisChoiceSystem(), observers=[observer], random_state=0)

# %%
# Slaves appear only when labour is plentiful
# -------------------------------------------
# With 1000 workers for about 750 jobs the mean wage should fall below the minimum wage at times, and workers should be
# paid below it, slave-days in Chesney's words. With 600 workers, fewer than the jobs, wages should rise to the top of
# the range and no worker should be underpaid.

scarce = simulate(AjanisChoiceSystem(n_workers=300), random_state=0)
fig, (wages, slaves) = plt.subplots(1, 2, figsize=(10, 4))
for name, run in (('1000 workers', trajectory), ('600 workers', scarce)):
    mean = [np.mean(run.agents('employers', frame).values['wage']) for frame in range(len(run.times))]
    wages.plot(run.times, mean, label=name)
    slaves.plot(run.times[1:], np.cumsum(run.values('day.slaves')[1:]), label=name)
wages.axhline(1.0, color='black', linestyle=':', label='minimum wage')
wages.set_xlabel('days')
wages.set_ylabel('mean wage')
wages.set_title('Wages of the employers')
wages.legend()
slaves.set_xlabel('days')
slaves.set_ylabel('slave-days')
slaves.set_title('Workers paid below the minimum wage')
slaves.legend()
fig.tight_layout()
plt.show()
