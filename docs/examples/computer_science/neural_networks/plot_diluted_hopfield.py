"""
# Asymmetric diluted Hopfield network

Four thousand neurons with sparse asymmetric couplings recall a stored pattern: started with an overlap of 0.3 with
the first of ten patterns, their overlap grows within a few parallel updates and settles near 0.92.
"""

# %%
# Every neuron reads about `K = 50` others, the network stores `p = 0.2 K = 10` patterns, and the inverse temperature
# is `beta = 3 / K`. The blueprint shows the overlap with the first pattern over 20 parallel updates.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb
from scipy import integrate

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'diluted_hopfield'
K, P, BETA = 50.0, 10, 3.0 / 50.0

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/overlap', name='overlap m', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, n_neurons=4000, connectivity=K, load=0.2, temperature_scale=3.0, overlap=0.3)
simulation = Simulation(
    system,
    dt=1,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(20)

# %%
# The annealed map of the overlap
# -------------------------------
# For large connectivity the local field along the pattern is Gaussian with mean `K m` and variance `K (p - m^2)`, so
# the overlap follows `m(t + 1) = f(m(t))` with `f(m) = E[tanh(beta (K m + sqrt(K (p - m^2)) z))]`, Eq. (6.55) of
# Bertin. The simulated overlaps should follow the iteration of `f`, and the pairs `(m(t), m(t + 1))` should lie on
# the curve `f`, reaching its stable fixed point where it crosses the diagonal.


def annealed(m):
    """Return the overlap after one parallel update in the annealed approximation."""
    spread = np.sqrt(K * (P - m**2))

    def integrand(z):
        return np.exp(-(z**2) / 2) * np.tanh(BETA * (K * m + spread * z)) / np.sqrt(2 * np.pi)

    return integrate.quad(integrand, -10, 10)[0]


overlaps = simulation.trajectory_['overlap']
predicted = [overlaps[0]]
for _ in overlaps[1:]:
    predicted.append(annealed(predicted[-1]))
grid = np.linspace(0, 1, 101)
fig, (series, return_map) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, overlaps, 'o', label='simulation')
series.plot(simulation.trajectory_.times, predicted, '--', color='black', label='annealed map')
series.set_xlabel('parallel update t')
series.set_ylabel('overlap m')
series.set_title('Recall of the first pattern')
series.legend()
return_map.plot(grid, [annealed(m) for m in grid], color='black', label='f(m)')
return_map.plot(grid, grid, ':', color='grey', label='m(t + 1) = m(t)')
return_map.plot(overlaps[:-1], overlaps[1:], 'o', label='simulation')
return_map.set_xlabel('m(t)')
return_map.set_ylabel('m(t + 1)')
return_map.set_title('Return map of the overlap')
return_map.legend()
fig.tight_layout()
plt.show()
