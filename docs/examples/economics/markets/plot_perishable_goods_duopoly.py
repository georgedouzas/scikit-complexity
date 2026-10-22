"""
# Perishable goods duopoly

The asymmetry `m = p_1 - p_2` of the shares of two shops of perishable goods grows from zero until one shop takes most
of the 200 customers: below the critical temperature the shop with more customers sells fresher goods and draws even
more customers.
"""

# %%
# The model is created by name from the catalog with half a product per customer in each shop and products that age
# over one unit of time, so the control parameter is `R = 1` and the critical temperature `T_c = R / (1 + R)^2 = 0.25`.
# The customers choose at the temperature 0.1. The observer saves every second frame with the default blueprint, whose
# primary view is the time series of `m`.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'perishable_goods_duopoly'
CONTROL = 1.0
CRITICAL = CONTROL / (1 + CONTROL) ** 2

system = create(MODEL, customers=200, stock=0.5, tau_a=1.0, T=0.1, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=300)

# %%
# Symmetry breaking against the temperature
# -----------------------------------------
# The mean-field equation of the representative customer, `dm/dt ~ T ln((1 - m) / (1 + m)) + h_1 - h_2` with the
# freshness `h_j = 1 / (1 + R / (1 +- m))`, has the stable root `m = 0` above `T_c` and a pair of roots `+-m*` below
# it, which grow continuously from zero since `R > sqrt(3) - 1`. The stationary asymmetry of runs at a range of
# temperatures should follow `m*`, slightly below it from the noise of the finite number of customers, and further
# below it just under `T_c`, where the slow growth of `m` has not ended within a run.


def mean_field_asymmetry(temperature):
    """Return the stable asymmetry of the mean-field equation at a temperature."""
    if temperature >= CRITICAL:
        return 0.0

    def drift(m):
        return temperature * np.log((1 - m) / (1 + m)) + 1 / (1 + CONTROL / (1 + m)) - 1 / (1 + CONTROL / (1 - m))

    return brentq(drift, 1e-6, 1 - 1e-12)


temperatures = np.linspace(0.3, 1.6, 8) * CRITICAL
asymmetries = []
for temperature in temperatures:
    model_system = create(MODEL, T=temperature, random_state=1)
    run = Simulation(model_system, random_state=1, observables=describe(MODEL).observables_of(model_system)).run(
        until=400,
    )
    asymmetries.append(np.abs(run.trajectory_['m'][-100:]).mean())
grid = np.linspace(0.2, 1.7, 200) * CRITICAL

fig, (series, transition) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, simulation.trajectory_['m'])
series.axhline(mean_field_asymmetry(0.1), color='black', linestyle='--', label='+-m* of mean field')
series.axhline(-mean_field_asymmetry(0.1), color='black', linestyle='--')
series.set_xlabel('time')
series.set_ylabel('asymmetry m')
series.set_title('One shop takes the customers at T = 0.1')
series.legend()
transition.plot(temperatures / CRITICAL, asymmetries, 'o', label='simulated')
transition.plot(grid / CRITICAL, [mean_field_asymmetry(value) for value in grid], color='black', label='mean field')
transition.set_xlabel('T / T_c')
transition.set_ylabel('stationary |m|')
transition.set_title('Continuous symmetry breaking at R = 1')
transition.legend()
fig.tight_layout()
plt.show()
