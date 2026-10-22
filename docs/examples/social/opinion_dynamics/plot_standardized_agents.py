"""
# Standardized agents

As the temperature rises and falls again, the share `q` of 100 complex agents that show the characteristic drops from
1 to near 0 and later jumps back, at a lower temperature, while the magnetization `m` of the standardized agents orders
only at the lowest temperatures: two discontinuous standardization transitions with hysteresis and a continuous
ordering.
"""

# %%
# The model is created by name from the catalog with `H = 50` configurations without the characteristic, the couplings
# `K = 10` and `J = 1`, and every agent standardized at the start, with a random spin. A schedule holds the
# temperature at 0.4 for 600 units of time, raises it linearly to 2.4 over 900 units, lowers it back to 0.4 over the
# next 900, and holds it again for 600. The observer saves every twelfth frame with the default blueprint, whose
# primary view is the time series of `q` and `m`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'standardized_agents'
K, J, H = 10.0, 1.0, 50
LOW, HIGH, HOLD, RAMP = 0.4, 2.4, 600.0, 900.0
PEAK, END = HOLD + RAMP, 2 * (HOLD + RAMP)


def temperature_at(time):
    """Return the temperature of the schedule at a time, held low, raised, lowered, and held low again."""
    return LOW + (HIGH - LOW) * float(np.clip(1 - abs(time - PEAK) / RAMP, 0, 1))


system = create(MODEL, agents=100, K=K, J=J, H=H, T=LOW, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=12)
schedule = {'standardization.T': temperature_at}
simulation = Simulation(
    system,
    observers=[observer],
    schedule=schedule,
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=END)

# %%
# Order parameters against the temperature
# ----------------------------------------
# The joint distribution of `q` and `m` takes the large deviation form `P ~ exp(N f(q, m))` with
# `f = -(1 - q) ln(1 - q) - (q - m) ln(q - m) / 2 - (q + m) ln(q + m) / 2 + K q^2 / 2T + J m^2 / 2T + (1 - q) ln H
# + q ln 2`. Its global maximum jumps from `q ~ 0` to `q ~ 1` at `T_S = K / (2 ln(H / 2))`, and `m` orders below
# `T_M = J`. The run should follow the branch it is on past `T_S` on both ramps, since the jump needs a fluctuation of
# the whole group, and should leave it at the spinodals, where the branch stops being a local maximum. On the
# standardized branch, `|m|` should approach the maximum of `f` at the low temperature and lag behind it on both
# ramps, since an agent changes the sign of its spin only when it proposes the opposite marked configuration, once in
# `H + 2` proposals.


def rate_function(q, m, temperature):
    """Return the rate function f(q, m) of the standardized agents at a temperature."""
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.nan_to_num((1 - q) * np.log(1 - q)) - np.nan_to_num((q - m) * np.log(q - m)) / 2
        entropy -= np.nan_to_num((q + m) * np.log(q + m)) / 2
    return entropy + (K * q**2 + J * m**2) / (2 * temperature) + (1 - q) * np.log(H) + q * np.log(2)


q_grid, m_grid = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401), indexing='ij')
inside = m_grid <= q_grid
temperatures = np.linspace(LOW, HIGH, 60)
maxima = []
for temperature in temperatures:
    values = np.where(inside, rate_function(q_grid, m_grid, temperature), -np.inf)
    best = np.unravel_index(np.argmax(values), values.shape)
    maxima.append((q_grid[best], m_grid[best]))
maxima = np.array(maxima)
times = simulation.trajectory_.times
ramp = np.array([temperature_at(time) for time in times])
rising = times <= PEAK

fig, (share, order) = plt.subplots(1, 2, figsize=(10, 4))
for axis, name in ((share, 'q'), (order, 'm')):
    values = np.abs(simulation.trajectory_[name])
    axis.plot(ramp[rising], values[rising], '.', markersize=3, label='temperature rising')
    axis.plot(ramp[~rising], values[~rising], '.', markersize=3, label='temperature falling')
    axis.plot(temperatures, maxima[:, 0 if name == 'q' else 1], color='black', label='maximum of f')
    axis.set_xlabel('temperature T')
    axis.legend()
share.axvline(K / (2 * np.log(H / 2)), color='grey', linestyle='--')
share.set_ylabel('share q of standardized agents')
share.set_title('Standardization with hysteresis around T_S')
order.axvline(J, color='grey', linestyle='--')
order.set_ylabel('magnetization |m|')
order.set_title('Ordering below T_M = J')
fig.tight_layout()
plt.show()
