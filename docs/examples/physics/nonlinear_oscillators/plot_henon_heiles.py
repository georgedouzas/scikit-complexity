"""
# Hénon-Heiles

A star orbiting in a triangular galactic potential: at low energy its orbit is a regular rosette, while at energy
`1 / 8` and above many orbits wander chaotically, and the Poincaré section shows islands of order in a sea of chaos.
"""

# %%
# The star starts on the line `x = 0` with the energy `E = 1 / 8`. The model logs the coordinates as time series only,
# so the observer below also logs the path of the star in the plane, with `y` negated because the 2D views of Rerun
# point their vertical axis down. The blueprint shows that path alone.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.analysis import poincare_section
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'henon_heiles'
ENERGY = 1 / 8


# Workaround: remove once the core draws planar trajectories y-up.
class OrbitObserver(RerunObserver):
    """Observer that also logs the path of the star in the plane."""

    def on_start(self, problem):
        super().on_start(problem)
        self.path_ = []

    def on_frame(self, frame):
        self.path_.append([frame.value('star.x'), -frame.value('star.y')])
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            self.recording_.log('orbit/path', rr.LineStrips2D([self.path_], radii=0.003, colors=[90, 170, 230]))
            self.recording_.log('orbit/star', rr.Points2D([self.path_[-1]], radii=0.02, colors=[250, 180, 60]))


def start(y, py, energy=ENERGY):
    """Return the Hénon-Heiles model started on `x = 0` at a given energy."""
    px = np.sqrt(2 * energy - py**2 - y**2 + 2 * y**3 / 3)
    return create(MODEL, x=0.0, y=y, px=px, py=py)


blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/orbit',
        name='orbit',
        visual_bounds=rrb.VisualBounds2D(x_range=(-0.65, 0.65), y_range=(-0.8, 0.5)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = OrbitObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=8)
simulation = Simulation(start(-0.1, 0.0), dt=0.05, observers=[observer]).run(until=100)

# %%
# Poincaré section
# ----------------
# Each orbit is recorded at its upward crossings of `x = 0`. Regular orbits lie on invariant tori and should trace
# closed curves around islands, while chaotic orbits should scatter over a connected sea between them.

fig, ax = plt.subplots(figsize=(6, 5))
for y in np.linspace(-0.3, 0.45, 12):
    section = poincare_section(start(y, 0.0), 'star.x', 0.0, until=2000)
    ax.plot(section['star.y'], section['star.y_d1'], '.', markersize=1.5)
ax.set_xlabel('y')
ax.set_ylabel('p_y')
ax.set_title('Poincaré section at x = 0, E = 1/8')
plt.show()
