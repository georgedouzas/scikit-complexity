"""
# Crop livestock household

A small farming family lives on its crop income and receives a grant to start rearing livestock. It buys animals
when it can afford them and its herd has room, sells a fifth of its herd every year, and the animals it buys can be
sold only after three years. Livestock raises the household's income, but only after the maturation delay.
"""

# %%
# The household starts with a wealth of 10000, four animals, and half a hectare of crops, for 12 years. The blueprint
# shows the wealth above the herd.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.resources import CropLivestockHouseholdSystem

SYSTEM = 'crop_livestock_household'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/household/wealth', name='wealth'),
        rrb.TimeSeriesView(origin='/household/livestock', name='livestock'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(CropLivestockHouseholdSystem(), observers=[observer])

# %%
# Incomes and the maturation delay
# --------------------------------
# The crop income stays at 80000 a year, just above the consumption of 75000. Animals bought in a year join the herd
# three years later, so the income from livestock should stay at zero for the first five years and then come in most
# years, lifting the wealth well above its start, as in Fig. 4.27 of Bala et al.

years = np.asarray(trajectory.times)
fig, (income, herd) = plt.subplots(1, 2, figsize=(10, 4))
income.step(years, trajectory.values('household.income_from_crop'), where='post', label='income from crop')
income.step(years, trajectory.values('household.income_from_livestock'), where='post', label='income from livestock')
income.step(years, trajectory.values('household.wealth'), where='post', label='wealth')
income.set_xlabel('years')
income.set_ylabel('money')
income.set_title('Incomes and wealth')
income.legend()
herd.step(years, trajectory.values('household.purchase'), where='post', label='animals bought')
herd.step(years, trajectory.values('household.sale'), where='post', label='animals sold')
herd.step(years, trajectory.values('household.livestock'), where='post', label='herd')
herd.set_xlabel('years')
herd.set_ylabel('animals')
herd.set_title('The herd')
herd.legend()
fig.tight_layout()
plt.show()
