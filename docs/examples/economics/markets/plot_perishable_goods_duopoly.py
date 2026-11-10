"""
# Perishable goods duopoly

The asymmetry `m = p_1 - p_2` of the shares of two shops of perishable goods grows from zero until one shop takes most
of the 200 customers: below the critical temperature the shop with more customers sells fresher goods and draws even
more customers.
"""

# %%
# The example builds `PerishableGoodsDuopolySystem` with half a product per customer in each shop and products that age
# over one unit of time, so the control parameter is `R = 1` and the critical temperature `T_c = R / (1 + R)^2 = 0.25`.
# The customers choose at the temperature 0.1. The two shops are the isolated nodes of a network, so the customers are
# not drawn in space: the blueprint shows the time series of `m` and, below it, the shares of customers at each shop,
# which the example computes from the trajectory and logs after the run.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.markets import PerishableGoodsDuopolySystem

SYSTEM = 'perishable_goods_duopoly'
CONTROL = 1.0
CRITICAL = CONTROL / (1 + CONTROL) ** 2
SHOPS = (0, 1)
COLORS = ((31, 119, 180), (255, 127, 14))


whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/m', name='asymmetry m', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(origin='/shares', name='shares of customers', axis_x=rrb.TimeAxis(view_range=whole_run)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = PerishableGoodsDuopolySystem(customers=200, stock=0.5, tau_a=1.0, T=0.1, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=2)
trajectory = simulate(system, observers=[observer], random_state=0)
shops = np.array([trajectory.get_agents('customers', frame).values['shop'] for frame in range(len(trajectory.times))])
for shop, color in zip(SHOPS, COLORS, strict=True):
    observer.recording_.log(
        f'shares/shop_{shop + 1}',
        rr.SeriesLines(colors=[color], names=f'shop {shop + 1}'),
        static=True,
    )
    observer.recording_.send_columns(
        f'shares/shop_{shop + 1}',
        indexes=[rr.TimeColumn('sim_time', duration=trajectory.times)],
        columns=rr.Scalars.columns(scalars=(shops == shop).mean(axis=1)),
    )
observer.recording_.flush()

# %%
# Symmetry breaking against the temperature
# -----------------------------------------
# The mean-field equation of the representative customer, `dm/dt ~ T ln((1 - m) / (1 + m)) + h_1 - h_2` with the
# freshness `h_j = 1 / (1 + R / (1 +- m))`, has the stable root `m = 0` above `T_c` and a pair of roots `+-m*` below
# it, which grow continuously from zero since `R > sqrt(3) - 1`. The stationary asymmetry of runs of 100 customers at a
# range of temperatures should follow `m*`, slightly below it from the noise of the finite number of customers, and
# further below it slightly under `T_c`, where the slow growth of `m` has not ended within a run.


def mean_field_asymmetry(temperature):
    """Return the stable asymmetry of the mean-field equation at a temperature."""
    if temperature >= CRITICAL:
        return 0.0

    def drift(m):
        return temperature * np.log((1 - m) / (1 + m)) + 1 / (1 + CONTROL / (1 + m)) - 1 / (1 + CONTROL / (1 - m))

    return brentq(drift, 1e-6, 1 - 1e-12)


temperatures = np.linspace(0.3, 1.6, 6) * CRITICAL
asymmetries = []
for temperature in temperatures:
    system_system = PerishableGoodsDuopolySystem(customers=100, T=temperature, random_state=1)
    run = simulate(system_system, random_state=1)
    asymmetries.append(np.abs(run.get_values('m')[-100:]).mean())
grid = np.linspace(0.2, 1.7, 200) * CRITICAL

fig, (series, transition) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times, trajectory.get_values('m'))
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
