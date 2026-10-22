"""
# Produce market

Shoppers with lists of fruit and vegetables walk round a market, visiting a stall they have not seen yet every step
and buying whatever on their list it sells. The lists empty geometrically, because every stall stocks only part of the
produce, while the money spent saturates at the cost of the lists at the markups of the stalls visited.
"""

# %%
# A thousand shoppers want between one and eight of twelve kinds of produce, and each of nine stalls sells six kinds
# at its own markup of 1 to 30 percent. The default blueprint shows the mean number of items left on the lists and the
# mean money spent.

from math import comb

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'produce_market'
N_SHOPPERS, N_STALLS, N_KINDS, MEAN_LIST = 1000, 9, 12, 4.5

system = create(MODEL, n_shoppers=N_SHOPPERS, n_stalls=N_STALLS, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=N_STALLS)

# %%
# Items left on the lists
# -----------------------
# A stall that sells `s` of the twelve kinds stocks any wanted item with probability `s / 12`, so after `t` visits an
# item is still wanted with probability `(1 - s / 12)^t` on average over the stocks, so the mean list of 4.5 items
# shrinks as `4.5 (1 - s / 12)^t` (dashed). The stocks of one market are fixed, though, and an item that `m` of the
# nine stalls do not sell survives `t` distinct visits with probability `C(m, t) / C(9, t)`. Summed over the lists,
# this exact expectation (solid) should match the simulation, including the floor left by kinds that no stall sells.

steps = np.arange(N_STALLS + 1)


def expected_left(trajectory):
    """Return the expected mean number of items left after every visit, for the stocks of the market."""
    stocks = trajectory.agents('stalls', 0).values['stock'].astype(int)
    lists = trajectory.agents('shoppers', 0).values['wanted'].astype(int)
    missing = np.array([sum(not stock >> kind & 1 for stock in stocks) for kind in range(N_KINDS)])
    wanted = np.array([[wish >> kind & 1 for kind in range(N_KINDS)] for wish in lists])
    survival = np.array([[comb(m, t) / comb(N_STALLS, t) for m in missing] for t in steps])
    return (wanted @ survival.T).mean(axis=0)


fig, (left, spent) = plt.subplots(1, 2, figsize=(10, 4))
for index, stocked in enumerate((3, 6, 9)):
    system = create(MODEL, n_shoppers=N_SHOPPERS, n_stalls=N_STALLS, n_items_stocked=stocked, random_state=0)
    trajectory = (
        Simulation(system, random_state=0, observables=describe(MODEL).observables_of(system))
        .run(until=N_STALLS)
        .trajectory_
    )
    color = f'C{index}'
    left.semilogy(steps, trajectory['items_left'], 'o', color=color, label=f'{stocked} kinds per stall')
    left.semilogy(steps, MEAN_LIST * (1 - stocked / N_KINDS) ** steps, '--', color=color, lw=0.8)
    left.semilogy(steps, expected_left(trajectory), color=color)
    spent.plot(steps, trajectory['spent'], 'o-', color=color, label=f'{stocked} kinds per stall')
left.set_ylim(1e-3, 10)
left.set_xlabel('stalls visited')
left.set_ylabel('mean items left')
left.set_title('Lists empty geometrically')
left.legend()
spent.set_xlabel('stalls visited')
spent.set_ylabel('mean money spent')
spent.set_title('Spending saturates')
spent.legend()
fig.tight_layout()
plt.show()
