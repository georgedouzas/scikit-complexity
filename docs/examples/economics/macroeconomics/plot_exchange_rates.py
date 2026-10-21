"""
# Exchange rates

Hamill and Gilbert's staged adjustment of a trading economy to a rise in its home prices. Prices move first and
trade volumes follow a stage later, so under a fixed exchange rate the balance of trade first swings into surplus,
the dearer exports still selling as before, and then into deficit once buyers turn away, the J-curve. A floating
rate depreciates at once instead, and the delayed fall of imports then carries the balance into surplus.
"""

# %%
# The economy starts in balance with the British trade volumes of 1999, and its home prices rise by 10 percent at the
# first stage. The exchange rate is fixed and both price elasticities of trade are `-1`. The blueprint shows the
# balance of trade over four stages.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'exchange_rates'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/balance', name='balance of trade', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, floating=False)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=4)

# %%
# Elasticities decide the sign
# ----------------------------
# Under a fixed rate the import prices do not move, so only exports respond: with the price rise `pi` and the export
# value `X0 PH0` of the start, the balance after adjustment is `X0 PH0 ((1 + pi) (1 + pi e_x) - 1)`, which turns
# negative once `|e_x| > 1 / (1 + pi)`, about 0.91. Under a floating rate the currency depreciates by the price rise at
# the first stage, export prices abroad stay put, and imports fall with `e_m`, so the balance becomes the surplus
# `-pi e_m X0 PH0 (1 + pi)`. The simulated balances should lie on both lines for every elasticity.

EXPORTS, PRICE, RISE = 32.0, 1.2, 0.1


def run(elasticity, floating):
    """Return the trajectory of two stages of adjustment with equal elasticities."""
    system = create(MODEL, floating=floating, export_elasticity=elasticity, import_elasticity=elasticity)
    return Simulation(system, dt=1).run(until=2).trajectory_


elasticities = -np.linspace(0.0, 1.5, 11)
grid = -np.linspace(0.0, 1.5, 100)
exact = {
    False: EXPORTS * PRICE * ((1 + RISE) * (1 + RISE * grid) - 1),
    True: -RISE * grid * EXPORTS * PRICE * (1 + RISE),
}
fig, (stages, sign) = plt.subplots(1, 2, figsize=(10, 4))
for floating, color in ((False, 'C0'), (True, 'C1')):
    name = 'floating' if floating else 'fixed'
    trajectory = run(-1.0, floating)
    stages.plot(trajectory.times, trajectory['balance'], 'o-', color=color, label=name)
    sign.plot(
        -elasticities,
        [run(elasticity, floating)['balance'][-1] for elasticity in elasticities],
        'o',
        color=color,
    )
    sign.plot(-grid, exact[floating], color=color, lw=0.8, label=name)
stages.axhline(0, color='grey', lw=0.8)
stages.set_xticks([0, 1, 2])
stages.set_xlabel('stage')
stages.set_ylabel('balance of trade')
stages.set_title('Elasticities of -1: a J-curve under a fixed rate')
stages.legend()
sign.axhline(0, color='grey', lw=0.8)
sign.axvline(1 / (1 + RISE), color='C3', linestyle='--', label='|e_x| = 1 / (1 + pi)')
sign.set_xlabel('size of both elasticities |e|')
sign.set_ylabel('balance after adjustment')
sign.set_title('Simulation (dots) and exact balance (lines)')
sign.legend()
fig.tight_layout()
plt.show()
