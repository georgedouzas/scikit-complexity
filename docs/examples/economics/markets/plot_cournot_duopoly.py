"""
# Cournot duopoly

Two firms choose their output every period as the best response to what they believe the rival will produce. Without
errors in the beliefs the outputs spiral into the Nash equilibrium, and noisy beliefs keep the total output fluctuating
around it.
"""

# %%
# The inverse demand is `P = a - b Q` with `a = 1500` and `b = 10`, the second firm has a marginal cost of 150, and the
# firms start from zero output. Their beliefs about the rival's last output are off by up to 5 percent. The blueprint
# shows the total output, which overshoots, settles on the Nash total of 95 units, and jitters around it.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cournot_duopoly'
A, B, COSTS = 1500.0, 10.0, (0.0, 150.0)
NASH = np.array([(A - 2 * COSTS[0] + COSTS[1]) / (3 * B), (A - 2 * COSTS[1] + COSTS[0]) / (3 * B)])

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/total', name='total output Q', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, outputs=(0.0, 0.0), costs=COSTS, a=A, b=B, error=0.05)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=60)

# %%
# Reaction curves and convergence
# -------------------------------
# Firm `i` best responds with `q_i = (a - c_i) / (2 b) - q_j / 2`, and the Nash equilibrium `q_i = (a - 2 c_i + c_j)
# / (3 b)` lies where the two reaction curves cross. Every simultaneous best response swaps the deviations of the two
# firms and halves them, so without errors the distance to the equilibrium should fall as `2^-t`.


def outputs(trajectory):
    """Return the outputs of the two firms, one row per frame."""
    return np.array([trajectory.agents('firms', frame).values['output'] for frame in range(len(trajectory.times))])


exact_system = create(MODEL, outputs=(100.0, 0.0), costs=COSTS, a=A, b=B)
exact = outputs(Simulation(exact_system, random_state=0).run(until=20).trajectory_)
noisy = outputs(simulation.trajectory_)
grid = np.linspace(0.0, 150.0, 2)
fig, (reaction, distance) = plt.subplots(1, 2, figsize=(10, 4))
reaction.plot((A - COSTS[0]) / (2 * B) - grid / 2, grid, color='C0', label='reaction of firm 1')
reaction.plot(grid, (A - COSTS[1]) / (2 * B) - grid / 2, color='C2', label='reaction of firm 2')
reaction.plot(noisy[:, 0], noisy[:, 1], '.', color='grey', ms=3, label='5% belief errors')
reaction.plot(exact[:, 0], exact[:, 1], 'o-', color='C1', ms=3, lw=0.8, label='exact beliefs')
reaction.plot(*NASH, '*', color='black', ms=12, label='Nash equilibrium')
reaction.set_xlim(0, 110)
reaction.set_ylim(-5, 80)
reaction.set_xlabel('output of firm 1')
reaction.set_ylabel('output of firm 2')
reaction.set_title('Best responses meet at the Nash equilibrium')
reaction.legend(fontsize=8)
steps = np.arange(len(exact))
distance.semilogy(steps, np.linalg.norm(exact - NASH, axis=1), 'o', color='C1', label='exact beliefs')
distance.semilogy(steps, np.linalg.norm(exact[0] - NASH) * 0.5**steps, color='black', lw=0.8, label='2^-t')
distance.set_xlabel('period t')
distance.set_ylabel('distance to the Nash equilibrium')
distance.set_title('Best responses halve the deviation')
distance.legend()
fig.tight_layout()
plt.show()
