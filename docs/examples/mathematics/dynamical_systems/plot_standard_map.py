"""
# Standard map

Above the critical kick `K ≈ 0.97` the phase plane of the kicked rotor splits into a chaotic sea, which orbits wander
over and diffuse through in momentum, and regular islands of invariant curves around stable periodic orbits, which
orbits never leave.
"""

# %%
# The model runs from the golden-mean torus at `K = 1.2` and logs the angle and the momentum as time series. Orbits
# from 40 starting points on a line across the plane, iterated for 400 steps each, are logged once as points in the
# plane of the angle and the momentum modulo 2π, and the blueprint puts this phase portrait first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'standard_map'


def orbits(K, starts, until):
    """Return the angle and the unwrapped momentum of the orbits from several starting points."""
    runs = [Simulation(create(MODEL, theta=theta, p=p, K=K), dt=1).run(until=until).trajectory_ for theta, p in starts]
    return np.array([run['map.theta'] for run in runs]), np.array([run['map.p'] for run in runs])


# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_portrait',
            name='phase portrait',
            visual_bounds=rrb.VisualBounds2D(x_range=[-5, 633], y_range=[-633, 5]),
        ),
        rrb.TimeSeriesView(origin='/map', name='angle and momentum', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(create(MODEL), dt=1, observers=[observer]).run(until=1000)
rng = np.random.default_rng(0)
starts = np.column_stack([np.linspace(0.1, 2 * np.pi - 0.1, 40), np.linspace(0.05, 2 * np.pi - 0.05, 40)])
theta, p = orbits(1.2, starts, until=400)
colors = np.repeat(rng.integers(60, 256, size=(40, 3)), theta.shape[1], axis=0)
points = np.column_stack([100 * theta.ravel(), -100 * np.mod(p, 2 * np.pi).ravel()])
observer.recording_.log('phase_portrait', rr.Points2D(points, radii=0.8, colors=colors), static=True)
observer.recording_.flush()

# %%
# Diffusion in momentum
# ---------------------
# Below the critical kick the invariant curves that span the angle should block the momentum, so the spread of an
# ensemble should stay bounded. Above it the spread should grow linearly in time, as a random walk, with the diffusion
# coefficient near the random-phase value `K² / 4` for large kicks. Close to the critical kick the walk should be much
# slower, since orbits stick to the remnants of the broken invariant curves: at `K = 1.2` the ensemble started near
# `p = 0` should not yet have spread in 300 steps.

ensemble = np.column_stack([rng.uniform(0, 2 * np.pi, 100), rng.uniform(0, 0.1, 100)])
fig, (portrait, spread) = plt.subplots(1, 2, figsize=(11, 4.5))
portrait.scatter(points[:, 0] / 100, -points[:, 1] / 100, s=0.2, c=colors / 255)
portrait.set_xlabel(r'angle $\theta$')
portrait.set_ylabel('momentum p mod 2π')
portrait.set_title('Chaotic sea and regular islands at K = 1.2')
for K in (0.8, 1.2, 2.5, 5.0):
    _, momenta = orbits(K, ensemble, until=300)
    variance = np.var(momenta - momenta[:, :1], axis=0)
    spread.loglog(np.arange(1, 301), variance[1:] / K**2, label=f"K = {K}")
spread.loglog([1, 300], [0.5, 150], color='black', linestyle='--', label='random phases $2 D t / K^2$')
spread.set_xlabel('step')
spread.set_ylabel(r'variance of $p$ / $K^2$')
spread.set_title('Bounded below the critical kick, diffusive above')
spread.legend(fontsize=8)
fig.tight_layout()
plt.show()
