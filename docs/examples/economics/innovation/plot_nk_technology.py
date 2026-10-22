"""
# NK technology

McNerney, Farmer, Redner, and Trancik's model of technological improvement: a technology is a set of components
whose costs depend on each other through a design structure matrix. Every unit of production is a chance to redesign
one component, which also changes the components that depend on it, and a redesign is kept only when it lowers the
total cost. The cost falls as a power of the cumulative production, Wright's law, and more slowly for designs whose
components are more interdependent.
"""

# %%
# Twenty components, each of whose redesigns also changes one other component (`d = 2`), are improved over 250 units
# of production. The blueprint shows the total cost, which falls in ever rarer steps as improvements get harder to
# find.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'nk_technology'
N_COMPONENTS, STEPS, SEEDS = 20, 3000, 8

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
system = create(MODEL, n_components=N_COMPONENTS, d=2, random_state=0)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=250)

# %%
# Wright's law
# ------------
# With component costs redrawn uniformly, McNerney and co-authors predict the total cost to fall as `t^-alpha` with
# `alpha = 1 / d` when every redesign changes `d` components. Averaged over eight designs, the cost should fall along
# a straight line on log-log axes, exactly at the slope `-1` for independent components, and with slopes that become
# shallower as `d` grows, approaching `-1 / d` only slowly.

fig, (curves, exponents) = plt.subplots(1, 2, figsize=(10, 4))
measured = []
for index, d in enumerate((1, 2, 3)):
    costs = np.mean(
        [
            Simulation(
                create(MODEL, n_components=N_COMPONENTS, d=d, random_state=seed),
                random_state=seed,
                observables=describe(MODEL).observables_of(system),
            )
            .run(until=STEPS)
            .trajectory_['total_cost']
            for seed in range(SEEDS)
        ],
        axis=0,
    )
    steps = np.arange(1, len(costs) + 1)
    curves.loglog(steps, costs, color=f'C{index}', label=f'd = {d}')
    curves.loglog(steps[100:], costs[100] * (steps[100:] / 100) ** (-1 / d), 'k:', lw=0.8)
    measured.append(-np.polyfit(np.log(steps[300:]), np.log(costs[300:]), 1)[0])
curves.set_xlabel('cumulative production t')
curves.set_ylabel('mean total cost')
curves.set_title('Power-law cost decline, t^(-1/d) dotted')
curves.legend()
exponents.plot([1, 2, 3], measured, 'o', label='fitted over t > 300')
exponents.plot([1, 2, 3], [1, 1 / 2, 1 / 3], 'k--', lw=0.8, label='1 / d')
exponents.set_xticks([1, 2, 3])
exponents.set_xlabel('components changed per redesign d')
exponents.set_ylabel('Wright exponent alpha')
exponents.set_title('Interdependence slows learning')
exponents.legend()
fig.tight_layout()
plt.show()
