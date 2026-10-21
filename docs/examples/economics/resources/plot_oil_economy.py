"""
# Oil economy

Meadows' oil company: capital extracts a nonrenewable resource and reinvests its profit, so the capital grows
exponentially while the oil is easy to reach. As the stock is drawn down the yield per unit of capital falls, profit
no longer covers depreciation, and the capital peaks and collapses, leaving oil in the ground. Finding more oil only
delays the peak by a few years.
"""

# %%
# The company starts with five units of capital and 1000 units of oil, sells at a price of 3, and reinvests up to 10
# percent of its capital a year, which lasts twenty years. The blueprint shows the capital, which rises, peaks, and
# declines over a century, together with the falling resource.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'oil_economy'
UNTIL = 100.0

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL), dt=1.0, observers=[observer]).run(until=UNTIL)

# %%
# More oil, a later peak
# ----------------------
# With investment capped at 10 percent of capital a year, the capital doubles about every `ln 2 / (0.1 - 1 / 20)`, 14
# years, while the oil is plentiful, and the extraction grows with it. Doubling the initial resource should therefore
# delay the peak of extraction by roughly one doubling time rather than double the lifetime of the industry, and the
# capital, which outlives the falling extraction for a while, should peak later still but shift even less.

fig, (capital, extraction) = plt.subplots(1, 2, figsize=(10, 4))
for index, resource in enumerate((1000.0, 2000.0, 4000.0)):
    trajectory = Simulation(create(MODEL, resource=resource), dt=0.5).run(until=UNTIL).trajectory_
    times, stock = trajectory.times, trajectory['oil.capital']
    peak = times[stock.argmax()]
    capital.plot(times, stock, color=f'C{index}', label=f'resource {resource:.0f}, peak at {peak:.0f}')
    rate = -np.gradient(trajectory['oil.resource'], times)
    extraction.plot(
        times,
        rate,
        color=f'C{index}',
        label=f'resource {resource:.0f}, peak at {times[rate.argmax()]:.0f}',
    )
capital.set_xlabel('year')
capital.set_ylabel('capital')
capital.set_title('Capital peaks and collapses')
capital.legend(fontsize=8)
extraction.set_xlabel('year')
extraction.set_ylabel('extraction rate')
extraction.set_title('Each doubling of oil buys about 16 years')
extraction.legend(fontsize=8)
fig.tight_layout()
plt.show()
