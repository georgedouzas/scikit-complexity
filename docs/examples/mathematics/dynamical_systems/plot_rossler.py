"""
# Rössler

The trajectory spirals out from the centre of the `x, y` plane, is lifted in a sudden spike of `z`, and is reinjected
near the centre: one band that is stretched and folded over itself at every turn, the simplest strange attractor.
"""

# %%
# The model logs its three states as time series. After the run, the whole trajectory is logged once more as a line in
# three dimensions, and the blueprint puts that view of the attractor first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.signal import find_peaks

from skcomplex.core.analysis import power_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'rossler'

camera = rrb.EyeControls3D(position=(0.0, -42.0, 18.0), look_target=(1.0, 0.0, 7.0), eye_up=(0.0, 0.0, 1.0))
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin='/attractor', name='attractor', eye_controls=camera, line_grid=False),
        rrb.TimeSeriesView(origin='/rossler', name='x, y, z', defaults=[rr.SeriesLines.from_fields(widths=1.5)]),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=25)
simulation = Simulation(create(MODEL), dt=0.05, observers=[observer]).run(until=300)
points = np.column_stack([simulation.trajectory_[f'rossler.{state}'] for state in 'xyz'])[400:]
observer.recording_.log('attractor', rr.LineStrips3D([points], radii=0.06, colors=[255, 170, 40]), static=True)
observer.recording_.flush()

# %%
# Return map of the maxima and spectrum
# -------------------------------------
# Over a longer run of 3000 time units, each maximum of `x` should be almost a function of the previous one, a
# single-humped curve like the logistic map, which is how the fold of the band makes the flow chaotic. The power
# spectrum of `x` should keep a sharp peak at the mean rotation frequency near 1 / 6 above a broad chaotic background.

x = Simulation(create(MODEL), dt=0.05).run(until=3000).trajectory_['rossler.x'][400:]
maxima = x[find_peaks(x)[0]]
frequencies, density = power_spectrum(x, dt=0.05)
fig, (returns, power) = plt.subplots(1, 2, figsize=(10, 4))
returns.plot(maxima[:-1], maxima[1:], '.', markersize=5)
returns.set_xlabel('maximum $x_n$')
returns.set_ylabel('next maximum $x_{n+1}$')
returns.set_title(f'Return map of {len(maxima)} maxima of x')
power.semilogy(frequencies, density)
power.axvline(1 / 6, color='black', linestyle=':', label='frequency 1 / 6')
power.set_xlim(0, 1)
power.set_ylim(density[frequencies < 1].min() / 2, 5 * density.max())
power.set_xlabel('frequency')
power.set_ylabel('power')
power.set_title('A sharp peak over a chaotic background')
power.legend()
fig.tight_layout()
plt.show()
