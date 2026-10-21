"""
# Baker map

When the two bands it stacks are thinner than the square, `lambda_a + lambda_b < 1`, the generalized baker map
contracts area, and the orbit settles on a strange attractor: horizontal lines stacked like a Cantor set in `y`,
gaps within gaps, uniform along `x`.
"""

# %%
# The map cuts the square at `alpha = 0.4` and squeezes the two pieces into bands of heights 0.25 and 0.4, and the
# model logs `x` and `y` as time series over 20000 steps. After the run, the whole orbit is logged once more as points
# in the square, and the blueprint puts this view of the attractor first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import kaplan_yorke_dimension, lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'baker_map'
ALPHA, LAMBDA_A, LAMBDA_B = 0.4, 0.25, 0.4

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/attractor',
            name='attractor',
            visual_bounds=rrb.VisualBounds2D(x_range=[-2, 102], y_range=[-102, 2]),
        ),
        rrb.TimeSeriesView(origin='/map', name='x, y'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, alpha=ALPHA, lambda_a=LAMBDA_A, lambda_b=LAMBDA_B)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=80)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=20000)
x, y = simulation.trajectory_['map.x'][20:], simulation.trajectory_['map.y'][20:]
observer.recording_.log(
    'attractor',
    rr.Points2D(np.column_stack([100 * x, -100 * y]), radii=0.15, colors=[90, 170, 255]),
    static=True,
)
observer.recording_.flush()

# %%
# Cantor set and dimension
# ------------------------
# The density of `y` should be a multifractal Cantor measure, with the gap between the bands `[0, 0.25]` and
# `[0.6, 1]` repeated at every scale. The Lyapunov exponents should be `-alpha ln alpha - (1 - alpha) ln (1 - alpha)`
# and `alpha ln lambda_a + (1 - alpha) ln lambda_b`, and the Kaplan-Yorke dimension they give should match the
# information dimension `1 + λ1 / |λ2|` of Farmer, Ott, and Yorke.

exponents = lyapunov_spectrum(system, until=20000)
expected = [
    -ALPHA * np.log(ALPHA) - (1 - ALPHA) * np.log(1 - ALPHA),
    ALPHA * np.log(LAMBDA_A) + (1 - ALPHA) * np.log(LAMBDA_B),
]
fig, (density, cantor) = plt.subplots(1, 2, figsize=(11, 4))
density.hist(y, bins=300, density=True)
density.set_xlabel('y')
density.set_ylabel('density')
density.set_title('Cantor measure along y')
cantor.plot(x, y, ',', alpha=0.5)
cantor.set_ylim(0.6, 0.76)
cantor.set_xlabel('x')
cantor.set_ylabel('y')
cantor.set_title(
    f'Zoom: Kaplan-Yorke dimension {kaplan_yorke_dimension(exponents):.3f}, '
    f'expected {1 + expected[0] / abs(expected[1]):.3f}',
)
fig.tight_layout()
plt.show()
