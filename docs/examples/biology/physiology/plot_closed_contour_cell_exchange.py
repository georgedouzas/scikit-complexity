"""
# Closed contour cell exchange

Eight cells in a closed ring each run the logistic map on their concentration of signalling molecules and take up a
share of the next cell's molecules. In a calm environment, `r = 2.5`, the cells start apart and synchronize within about
ten steps to one common concentration, an equilibrium inside the stability domain of Mihailović et al. The view
shows the eight concentrations converging onto one line.
"""

# %%
# The model is created by name from the catalog with 8 cells, the logistic parameter `r = 2.5`, the coupling
# `c = 0.1`, and the affinity `p = 0.5`, with concentrations drawn between 0.2 and 0.8. The run lasts 40 steps, and
# the blueprint shows the concentrations of all cells over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'closed_contour_cell_exchange'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/cells',
        name='concentrations',
        axis_x=rrb.TimeAxis(view_range=whole_run),
        defaults=[rr.SeriesLines.from_fields(widths=2.0)],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, r=2.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(40)

# %%
# Stability domain and desynchronization
# --------------------------------------
# An equilibrium is stable for every coupling when it lies in `((r - 1) / 2r, (r + 1) / 2r)` and above
# `p^(1 / (1 - p)) = 0.25`. Swept through `r`, the synchronized equilibrium should stay inside the shaded domain while
# the cells agree, and the cells should split onto many values once the logistic map turns chaotic near `r = 3.6`.

rates = np.linspace(2.0, 4.0, 41)
fig, (diagram, spread) = plt.subplots(1, 2, figsize=(10, 4))
spreads = []
for rate in rates:
    late = Simulation(create(MODEL, r=rate), dt=1).run(200).trajectory_['cells.x'][-30:]
    diagram.plot(np.full(late.size, rate), late.ravel(), '.', markersize=2, color='tab:blue')
    spreads.append(np.ptp(late, axis=1).max())
diagram.fill_between(rates, (rates - 1) / (2 * rates), (rates + 1) / (2 * rates), color='tab:green', alpha=0.2)
diagram.set_xlabel('logistic parameter r')
diagram.set_ylabel('concentration of the cells, last 30 steps')
diagram.set_title('Synchronized equilibria lie in the stability domain')
spread.semilogy(rates, np.maximum(spreads, 1e-16), 'o-', markersize=3)
spread.set_xlabel('logistic parameter r')
spread.set_ylabel('largest spread between cells')
spread.set_title('The cells desynchronize in a chaotic environment')
fig.tight_layout()
plt.show()
