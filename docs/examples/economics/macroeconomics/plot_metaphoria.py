"""
# Metaphoria

A mortal population lives on patches of food that resist harvesting. Hunter-gatherers pool their labour on a patch and
share the harvest, and they have children who inherit, with mutations, their propensities to breed and to explore, so
the propensities evolve under the test of survival. The population neither dies out nor explodes. The view shows the
population of the hunter-gatherers over 250 steps, rising from 100 in a first wave of births and then fluctuating
between about 80 and 150, above the births and the starvations of every step.
"""

# %%
# The system is built from its class with 100 adults on a periodic 20 x 20 lattice of patches whose food capacity is
# drawn up to 15 and whose resistance up to 5. Every adult brings 5 units of labour and eats 2 units of food per step.
# The blueprint shows the observable `population` above the reporters `census.births` and `census.starvations`.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.macroeconomics import MetaphoriaSystem

SYSTEM = 'metaphoria'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/population', name='population', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(
            origin='/reporters',
            contents=['/reporters/census.births', '/reporters/census.starvations'],
            name='births and starvations',
            axis_x=rrb.TimeAxis(view_range=whole_run),
        ),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = MetaphoriaSystem(n_people=100, mode='hunter_gatherer', random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(system, until=250, observers=[observer], random_state=0)

# %%
# Hunter-gatherers against a monetary economy
# -------------------------------------------
# Gooding (2019) ran both organisations on the same land with the same people. In the monetary economy firms harvest
# with paid workers, sell food in shops, and borrow their wages from a bank that creates money as loans. On land where
# the hunter-gatherers thrive, the monetary population should dwindle towards extinction. With abundant food, 50 units
# per patch and no resistance, it should live on while people still starve every step, and money should concentrate,
# with a Gini coefficient above the 0.5 that Gooding reports. The money of all holders, the bank's issued money
# included, should sum to zero.

monetary = MetaphoriaSystem(n_people=100, mode='monetary', random_state=0)
scarce = simulate(monetary, until=150, random_state=0)
rich = MetaphoriaSystem(n_people=100, mode='monetary', max_food=50.0, max_resistance=0.0, random_state=0)
abundant = simulate(rich, until=150, random_state=0)
holders = [abundant.get_agents(group, 150).values for group in ('people', 'businesses', 'bank')]
net_money = sum(np.sum(values['money']) for values in holders)

fig, (populations, money) = plt.subplots(1, 2, figsize=(11, 4))
populations.plot(trajectory.times[:151], trajectory.get_values('population')[:151], label='hunter-gatherers')
populations.plot(scarce.times, scarce.get_values('population'), label='monetary')
populations.plot(abundant.times, abundant.get_values('population'), label='monetary, abundant food')
populations.set_xlabel('step')
populations.set_ylabel('population')
populations.set_title('Money dwindles where gatherers thrive')
populations.legend()
money.plot(abundant.times, abundant.get_values('money_gini'), label='Gini coefficient of money')
starved = np.nan_to_num(abundant.get_values('census.starvations')) / np.maximum(abundant.get_values('population'), 1)
money.plot(abundant.times, starved, label='starvations per person')
money.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Gini of 0.5 (Gooding)')
money.set_xlabel('step')
money.set_ylabel('share')
money.set_title(f'Abundant food, net money {net_money:.1e}')
money.legend()
fig.tight_layout()
plt.show()
