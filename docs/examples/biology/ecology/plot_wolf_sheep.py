"""
# Wolf and sheep

Sheep graze and wolves hunt the sheep on a field of regrowing grass. The populations rise and fall in turn, and the
grass is eaten down to dark patches where the sheep gather.
"""

# %%
# The three groups share one 2D view of the 30 by 30 grid: grass from eaten (purple) to grown (yellow), sheep in white,
# and wolves in red. The time series below it count the sheep and the wolves, which oscillate in turn.

import matplotlib.pyplot as plt
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'wolf_sheep'

whole = rrb.TimeRangeBoundary.infinite()
agents = rrb.Spatial2DView(
    origin='/',
    contents=['/grass/**', '/sheep/**', '/wolves/**'],
    name='field',
    overrides={
        'grass': rr.Points2D.from_fields(radii=0.3).visualizer(),
        'sheep': rr.Points2D.from_fields(colors=[240, 240, 240], radii=0.45).visualizer(),
        'wolves': rr.Points2D.from_fields(colors=[230, 50, 50], radii=0.6).visualizer(),
    },
)
counts = rrb.TimeSeriesView(
    origin='/reporters',
    contents=['/reporters/sheep.count', '/reporters/wolves.count'],
    name='sheep and wolves',
    defaults=[rr.SeriesLines.from_fields(widths=2.5)],
    overrides={
        'reporters/sheep.count': rr.SeriesLines.from_fields(colors=[240, 240, 240]).visualizer(),
        'reporters/wolves.count': rr.SeriesLines.from_fields(colors=[230, 50, 50]).visualizer(),
    },
    axis_x=rrb.TimeAxis(view_range=rrb.TimeRange(start=whole, end=whole)),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(agents, counts, row_shares=[2, 1]),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, width=30, height=30, sheep=200, wolves=20, regrowth=20.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=300)

# %%
# Populations over time
# ---------------------
# The counts should oscillate out of phase, as in the Lotka-Volterra model: wolves increase after the sheep peak,
# and the sheep recover after the wolves decline. The grass, regrowing in 20 steps, keeps the sheep from growing
# without bound.

times = simulation.trajectory_.times
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times, simulation.trajectory_['sheep.count'], label='sheep')
ax.plot(times, simulation.trajectory_['wolves.count'], label='wolves', color='tab:red')
ax.set_xlabel('step')
ax.set_ylabel('number of agents')
ax.set_title('Sheep and wolves oscillate in turn')
ax.legend()
plt.show()
