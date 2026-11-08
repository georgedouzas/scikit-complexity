"""
# Order book with gurus

Noise traders place market and limit orders around random forecasts in an order book, but every day each adopts the
forecast of the trader it follows, and it moves its link towards wealthier traders. Successful traders become gurus
with many followers, the forecasts synchronize, and the returns of the market get fat tails and bursts of volatility.
"""

# %%
# A hundred traders with cash 1000 and 100 stocks each make 100 intraday operations a day for 300 days. The default
# blueprint shows the price, the returns, and the followers of the guru.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.finance import GuruOrderBookSystem

SYSTEM = 'guru_order_book'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulate(GuruOrderBookSystem(), observers=[observer])
trajectory = simulate(GuruOrderBookSystem(), until=1000)

# %%
# Coordination and fat tails
# --------------------------
# Periods in which the guru gathers many followers should come with large returns, and the distribution of the daily
# returns should have a much larger kurtosis than a Gaussian, as Caiani et al. find for coordination periods.

returns = trajectory.values('market.log_return')[1:]
followers = trajectory.values('market.guru_followers')[1:]
fig, (series, tails) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times[1:], returns, lw=0.6, label='log return')
series.plot(trajectory.times[1:], followers / 1000, lw=0.8, label='followers of the guru / 1000')
series.set_xlabel('day')
series.set_title('Returns and coordination')
series.legend()
standardized = (returns - returns.mean()) / returns.std()
tails.hist(standardized, bins=60, density=True, log=True, label=f'excess kurtosis {kurtosis(returns):.0f}')
grid = np.linspace(standardized.min(), standardized.max(), 200)
tails.plot(grid, np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi), 'k--', lw=0.8, label='Gaussian')
tails.set_xlabel('standardized daily return')
tails.set_ylabel('density')
tails.set_title('Fat tails of the returns')
tails.legend()
fig.tight_layout()
plt.show()
