"""
# Gooding double auction

Buyers who value a good at about 120 start bidding near the sellers' cost of about 20, and sellers start asking near
the buyers' value. They move their quotes towards each other until they trade, and then try to push the price their
way. In the short run the price settles at the competitive level, halfway between value and cost.
"""

# %%
# A hundred sellers with five units each meet a hundred buyers who want five units each, for 300 iterations. The
# default blueprint shows the traded price, the volume, and the mean bid and ask.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.markets import GoodingDoubleAuctionSystem

SYSTEM = 'gooding_double_auction'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
short = simulate(GoodingDoubleAuctionSystem(), observers=[observer])
long = simulate(GoodingDoubleAuctionSystem(n_buyers=10, n_sellers=10, unlimited=True), until=5000)

# %%
# Short and long run
# ------------------
# The bid and the ask should meet near 70 and the 500 units should trade there, as in Fig. 11.6 of Gooding. With
# unlimited supply and demand, the price keeps moving between the cost and the value instead of resting at the
# equilibrium.

fig, (quotes, drift) = plt.subplots(1, 2, figsize=(10, 4))
times = np.asarray(short.times)
quotes.plot(times, short.values('auction.mean_bid'), label='mean bid')
quotes.plot(times, short.values('auction.mean_ask'), label='mean ask')
quotes.plot(times, short.values('auction.price'), label='last price')
quotes.axhline(70, color='black', linestyle=':', lw=0.8)
quotes.set_xlabel('iteration')
quotes.set_ylabel('price')
quotes.set_title('Short run with limited units')
quotes.legend()
drift.plot(np.asarray(long.times), long.values('auction.price'), lw=0.6)
drift.axhline(70, color='black', linestyle=':', lw=0.8)
drift.set_xlabel('iteration')
drift.set_ylabel('last price')
drift.set_title('Long run with unlimited units')
fig.tight_layout()
plt.show()
