"""
# Sprott jerk

The jerk equation `x''' = -A x'' - x' + |x| - 1` has a single nonlinearity, an absolute value, yet at `A = 0.6` its
trajectory in the space of position, velocity, and acceleration never settles: it loops around a folded band, a
strange attractor of one of the simplest chaotic flows.
"""

# %%
# The run lasts 300 time units from rest with unit acceleration, and the model logs `x` and its two derivatives as time
# series. After the run, the trajectory in `(x, x', x'')` is logged once more as a line in three dimensions, and the
# blueprint puts this view of the attractor first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.signal import find_peaks

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sprott_jerk'

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='attractor', line_grid=False),
        rrb.TimeSeriesView(origin='/jerk', name='x and derivatives'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=12)
simulation = Simulation(create(MODEL), dt=0.1, observers=[observer]).run(until=300)
trajectory = simulation.trajectory_
points = np.column_stack([trajectory[name] for name in ('jerk.x', 'jerk.x_d1', 'jerk.x_d2')])[200:]
observer.recording_.log('attractor', rr.LineStrips3D([points], radii=0.01, colors=[255, 170, 40]), static=True)
observer.recording_.flush()

# %%
# Return map and Lyapunov exponents
# ---------------------------------
# Successive maxima of `x` should lie close to a single curve, the trace of the fold of the band. The largest Lyapunov
# exponent should be positive, near the 0.036 that Sprott reports, and the exponents should sum to the divergence `-A`.

x = Simulation(create(MODEL), dt=0.05).run(until=3000).trajectory_['jerk.x'][2000:]
maxima = x[find_peaks(x)[0]]
# The symbolic Jacobian of `|x|` divides by `|x|`, so the tangent dynamics start off `x = 0` (see fix-core-defects).
exponents = lyapunov_spectrum(create(MODEL, x=0.1), until=3000)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(maxima[:-1], maxima[1:], '.', markersize=4)
ax.set_xlabel('maximum $x_n$')
ax.set_ylabel('next maximum $x_{n+1}$')
ax.set_title(f'Return map; exponents {exponents[0]:.3f}, {exponents[1]:.3f}, {exponents[2]:.3f}')
plt.show()
