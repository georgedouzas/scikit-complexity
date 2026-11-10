"""
# Fishery with measurement errors

A fish stock grows logistically with a random growth rate, and the fishermen take the quota `x~ - K / 2` from their
measurement `x~` of the stock. The rule pulls the stock towards `K / sqrt(2)`, but the noise of the growth and the
errors of the measurement can push a stock to extinction.
"""

# %%
# Two hundred independent fisheries start at half the carrying capacity, with `R0 = 1`, `sigma_R = 1.4`, and a
# measurement error of standard deviation 0.2, for 5000 steps of length 0.01. The default blueprint shows the mean
# stock, the mean quota, and the share of extinct stocks.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.resources import FisheryMeasurementErrorsSystem

SYSTEM = 'fishery_measurement_errors'
STEPS = 5000


def run(**settings):
    """Return the trajectory of the fisheries with some settings."""
    return simulate(FisheryMeasurementErrorsSystem(sigma_r=1.4, **settings), until=STEPS, random_state=0)


observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
trajectory = simulate(
    FisheryMeasurementErrorsSystem(sigma_r=1.4, error=0.2),
    until=STEPS,
    observers=[observer],
    random_state=0,
)

# %%
# Extinction and the size of the errors
# -------------------------------------
# Without noise the stock settles at `K / sqrt(2)` and the quota at `(sqrt(2) - 1) K / 2`, below the largest
# sustainable harvest `R0 K / 4` at `K / 2`. With a quota floored at zero, larger measurement errors make the
# fishermen fish stocks that are already low, so the share of extinct stocks after the run should grow with the error.
# Letting a negative quota restock the fishery, as the formula of the source does, keeps every stock alive.

errors = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
floored = [run(error=error).get_values('extinct')[-1] for error in errors]
restocked = [run(error=error, restocking=True).get_values('extinct')[-1] for error in errors]
fig, (stock, extinction) = plt.subplots(1, 2, figsize=(10, 4))
stock.plot(trajectory.times * 0.01, trajectory.get_values('mean_stock'), label='mean stock')
stock.plot(trajectory.times * 0.01, trajectory.get_values('mean_catch'), label='mean quota')
stock.axhline(2**-0.5, color='black', linestyle='--', lw=0.8, label='K / sqrt(2)')
stock.set_xlabel('time')
stock.set_ylabel('stock and quota per unit of K')
stock.set_title('Stocks and quotas with errors of 0.2')
stock.legend()
extinction.plot(errors, floored, 'o-', label='quota floored at zero')
extinction.plot(errors, restocked, 's--', label='negative quota restocks')
extinction.set_xlabel('standard deviation of the measurement error')
extinction.set_ylabel('share of extinct stocks at t = 50')
extinction.set_title('Extinction against the measurement error')
extinction.legend()
fig.tight_layout()
plt.show()
