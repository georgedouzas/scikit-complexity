"""
# Grain storage system

A retailer keeps a grain stock to smooth prices for consumers. Grain on order arrives after a receiving delay, the stock
releases a fixed fraction every week, and the retailer orders to close the gap to a desired stock, ignoring the grain
already on order. The supply line overshoots, and the stock, the receiving rate, and the procurement oscillate before
they settle.
"""

# %%
# The system is created by name from the catalog with the values of Bala, Arshad and Noh: an initial stock of 1000
# tonnes and 10000 tonnes on order, a desired stock of 6000 tonnes, a release fraction of 0.025 per week, a receiving
# delay of 10 weeks, and an adjustment delay of 5 weeks. The run lasts 100 weeks, and the default blueprint shows the
# two stocks and the three flows.

import matplotlib.pyplot as plt

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'grain_storage'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run()

# %%
# Oscillation from the supply line
# --------------------------------
# The stock should rise far above the desired stock, fall below its final value, and settle at
# `6000 / (1 + 0.025 * 5)`, about 5333 tonnes, where the release balances the receiving. The procurement turns
# negative while the stock exceeds the desired one.

trajectory = simulation.trajectory_
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
top.plot(trajectory.times, trajectory.values('store.stock'), label='grain stock')
top.axhline(6000, linestyle=':', color='black', label='desired grain stock')
top.set_ylabel('tonnes')
top.legend()
bottom.plot(trajectory.times, trajectory.values('store.procurement'), label='procurement rate')
bottom.plot(trajectory.times, trajectory.values('store.receiving'), label='receiving rate')
bottom.set_xlabel('weeks')
bottom.set_ylabel('tonnes per week')
bottom.legend()
plt.show()
