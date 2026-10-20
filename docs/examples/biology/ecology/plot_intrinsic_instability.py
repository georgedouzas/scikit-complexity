"""
# Intrinsic instability

The resident population fluctuates about 1 until a rare mutation seeds an invader. The invader grows exponentially and
drives the resident population to collapse, although no parameter of the model changes.
"""

# %%
# The populations start at `(1.1, 0)` with the parameters of Jensen's Figure 13.2, `eta = 0.1`,
# `alpha = beta = nu = 0.01`, and a mutation probability of 0.001 per step, and evolve for 2000 steps. The blueprint
# shows the resident population above the invader.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'intrinsic_instability'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/populations/x', name='resident x', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(origin='/populations/y', name='invader y', axis_x=rrb.TimeAxis(view_range=whole_run)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=10, observers=[observer], random_state=0).run(2000)

# %%
# Leaving the fixed point along its unstable direction
# ----------------------------------------------------
# The fixed point `(1, 0)` of the deterministic part has the eigenvalues `-eta` along the resident axis and `beta`
# along `(-1, (eta + beta) / (alpha X))`. Once seeded, the invader should grow as `(1 + beta)^t`, a straight line of
# slope `log(1 + beta)` on a log scale, and the trajectory in the `xy`-plane should leave the axis `y = 0` and turn
# towards `x = 0`, as in Jensen's Figure 13.2.

trajectory = simulation.trajectory_
times, x, y = np.asarray(trajectory.times), trajectory['populations.x'], trajectory['populations.y']
seeded = int(np.argmax(y > 0))
growth = y[seeded] * (1 + 0.01) ** (times[seeded:] - times[seeded])
fig, (plane, invader) = plt.subplots(1, 2, figsize=(10, 4))
plane.plot(x[seeded:], y[seeded:])
plane.plot(x[:seeded], np.full(seeded, 1e-3), color='grey', label='before the mutation, y = 0')
plane.set_yscale('log')
plane.set_xlabel('resident x')
plane.set_ylabel('invader y')
plane.set_title('The trajectory leaves the fixed point (1, 0)')
plane.legend()
invader.semilogy(times[seeded:], y[seeded:], label='simulation')
invader.semilogy(times[seeded:], growth, '--', color='black', label='(1 + beta)^t')
invader.set_xlabel('step t')
invader.set_ylabel('invader y')
invader.set_title('Exponential growth of the invader')
invader.legend()
fig.tight_layout()
plt.show()
