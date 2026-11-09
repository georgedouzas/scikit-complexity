"""
# Resistance to three drugs

A single drug wipes out the sensitive cells within days, and the cells resistant to it, which the tumour seeded by
mutation, grow back into a tumour of 10^11 cells.
"""

# %%
# The system is built from its class with drug 1 infused at a constant rate of 10 dosage units a day and the two other
# drugs withheld, from a tumour of 4.6 x 10^11 sensitive cells. The observer saves every second frame with the default
# blueprint, whose primary view is the time series of the populations, the drugs, and their toxicities.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import MultidrugResistanceSystem

SYSTEM = 'multidrug_resistance'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(MultidrugResistanceSystem(u2=0.0, u3=0.0), dt=2.0, observers=[observer], until=600)

# %%
# One drug against three
# ----------------------
# Under one drug the cells resistant to it should regrow at nearly the rate `r = 0.0099` a day. Under all three drugs
# every population sensitive to at least one drug should vanish, and only the few cells resistant to all three should
# survive, so the tumour should stay orders of magnitude smaller.

names = ('S', 'R1', 'R2', 'R3', 'R12', 'R13', 'R23', 'R123')
single = trajectory
combined = simulate(MultidrugResistanceSystem(), dt=2.0, until=600)
fig, (one, three) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axis, trajectory, title in ((one, single, 'Drug 1 alone'), (three, combined, 'All three drugs')):
    for name in ('S', 'R1', 'R12', 'R123'):
        axis.semilogy(trajectory.times, np.maximum(trajectory.get_values(f'tumour.{name}'), 1e-3), label=name)
    axis.semilogy(
        trajectory.times,
        sum(trajectory.get_values(f'tumour.{name}') for name in names),
        'k--',
        label='total',
    )
    axis.set_ylim(1e-3, 1e13)
    axis.set_xlabel('time (days)')
    axis.set_title(title)
    axis.legend()
one.set_ylabel('cells')
fig.tight_layout()
plt.show()
