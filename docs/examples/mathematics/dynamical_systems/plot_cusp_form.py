"""
# Cusp normal form

As the asymmetry `beta1` is swept slowly up and back down inside the cusp, the state `x` follows one branch of
equilibria until the branch folds away, then jumps to the other branch, and on the way back it jumps at the opposite
fold: a hysteresis loop, the catastrophe of the cusp.
"""

# %%
# With `beta2 = 1` the system is bistable for `|beta1| < 2 / (3√3) ≈ 0.385`. The schedule moves `beta1` linearly from
# -0.6 to 0.6 and back over 600 time units, slowly compared with the relaxation time of order 1. The default blueprint
# with the parameters shows `x` and `beta1` as time series.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'cusp_form'
PERIOD = 600.0


def beta1_at(time):
    """Return the asymmetry along a triangular sweep from -0.6 to 0.6 and back."""
    phase = (time % PERIOD) / PERIOD
    return -0.6 + 2.4 * phase if phase < 0.5 else 1.8 - 2.4 * phase


system = create(MODEL, x=-1.0, beta1=-0.6, beta2=1.0)
observer = RerunObserver(
    MODEL,
    save_path=f'{MODEL}.rrd',
    blueprint=default_blueprint(system, parameters=True),
    interval=3,
)
simulation = Simulation(system, dt=1.0, observers=[observer], schedule={'form.beta1': beta1_at}).run(until=PERIOD)

# %%
# Hysteresis loop
# ---------------
# Plotted against `beta1`, the run should trace the outer stable branches of the equilibria `beta1 + x - x³ = 0`, jump
# up near the fold at `beta1 = +0.385` and down near `beta1 = -0.385`, a little past the folds, a delay of slow passage
# that shrinks with the sweep rate, and never visit the unstable middle branch.

trajectory = simulation.trajectory_
fine = Simulation(create(MODEL, x=-1.0, beta1=-0.6, beta2=1.0), dt=0.1, schedule={'form.beta1': beta1_at})
fine = fine.run(until=PERIOD).trajectory_
x = np.linspace(-1.3, 1.3, 400)
branch = x**3 - x
fig, (loop, series) = plt.subplots(1, 2, figsize=(11, 4))
loop.plot(branch[np.abs(x) >= 1 / np.sqrt(3)], x[np.abs(x) >= 1 / np.sqrt(3)], '.', color='black', markersize=1)
loop.plot(branch[np.abs(x) < 1 / np.sqrt(3)], x[np.abs(x) < 1 / np.sqrt(3)], '--', color='gray', label='unstable')
loop.plot([beta1_at(time) for time in fine.times], fine['form.x'], color='C3', label='sweep')
loop.axvline(2 / (3 * np.sqrt(3)), color='C0', linestyle=':', label='folds')
loop.axvline(-2 / (3 * np.sqrt(3)), color='C0', linestyle=':')
loop.set_xlim(-0.65, 0.65)
loop.set_xlabel(r'$\beta_1$')
loop.set_ylabel('x')
loop.set_title('Hysteresis between the two folds')
loop.legend(fontsize=8)
series.plot(trajectory.times, trajectory['form.x'], label='x')
series.plot(trajectory.times, [beta1_at(time) for time in trajectory.times], label=r'$\beta_1$')
series.set_xlabel('time')
series.set_ylabel('value')
series.set_title('Jumps between the branches')
series.legend()
fig.tight_layout()
plt.show()
