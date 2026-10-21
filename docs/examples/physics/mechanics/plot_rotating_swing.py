"""
# Rotating swing

The chairs of a merry-go-round swing outwards as the stand turns. Thrown out past their equilibrium angle they swing
back and forth about it, and since the stand conserves angular momentum it slows down whenever the chairs fly out and
speeds up whenever they fall in.
"""

# %%
# Eight chairs hang on chains of length 1 from rods of length 1 on a stand of inertia 10 that starts turning at 2
# radians per unit time, with the chains at 0.1 radians. The stand angle grows without bound and lies in `[0, 2π)`, so
# the observer below logs only the chain angle and the angular velocity of the stand, which oscillate in opposition.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'rotating_swing'


# Workaround: remove once the core draws angles wrapped to (-π, π] and lets a blueprint leave the stand angle out.
class SwingObserver(RerunObserver):
    """Observer that also logs the chain angle and the angular velocity of the stand."""

    def on_start(self, problem):
        super().on_start(problem)
        for name, color in (('chain angle', [250, 150, 60]), ('stand speed', [90, 170, 230])):
            path = 'chairs/' + name.replace(' ', '_')
            self.recording_.log(path, rr.SeriesLines(colors=color, names=name, widths=2), static=True)

    def on_frame(self, frame):
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            self.recording_.log('chairs/chain_angle', rr.Scalars(frame.value('swing.theta')))
            self.recording_.log('chairs/stand_speed', rr.Scalars(frame.value('swing.phi_d1')))


blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/chairs', name='chairs'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, omega=2.0, theta=0.1)
observer = SwingObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=20)

# %%
# Angular momentum and the equilibrium angle
# ------------------------------------------
# Without torque or friction the angular momentum `[J + n m (a + l sin θ)^2] φ'` should stay at its initial value. The
# chain angle should oscillate about the steady angle at which the chains and the stand turn together with that
# momentum, the solution of `g tan θ = φ'^2 (a + l sin θ)` with `φ' = L / [J + n m (a + l sin θ)^2]`, dashed.

times = np.asarray(simulation.trajectory_.times)
theta = np.ravel(simulation.trajectory_['swing.theta'])
speed = np.ravel(simulation.trajectory_['swing.phi_d1'])
momentum = (10.0 + 8 * (1 + np.sin(theta)) ** 2) * speed
grid = np.linspace(0.0, 1.5, 3000)
steady_speed = momentum[0] / (10.0 + 8 * (1 + np.sin(grid)) ** 2)
equilibrium = grid[np.argmin(np.abs(9.81 * np.tan(grid) - steady_speed**2 * (1 + np.sin(grid))))]
fig, (angle, conserved) = plt.subplots(1, 2, figsize=(10, 4))
angle.plot(times, theta, label='chain angle')
angle.axhline(equilibrium, linestyle='--', color='black', label='steady angle at the conserved momentum')
angle.set_xlabel('time')
angle.set_ylabel('θ (radians)')
angle.set_title('Chains swing about the equilibrium angle')
angle.set_ylim(0.0, 0.75)
angle.legend(loc='upper center', ncol=2, fontsize=8)
conserved.plot(times, (momentum - momentum[0]) / momentum[0])
conserved.set_xlabel('time')
conserved.set_ylabel('relative change of the angular momentum')
conserved.set_title('Angular momentum is conserved')
fig.tight_layout()
plt.show()
