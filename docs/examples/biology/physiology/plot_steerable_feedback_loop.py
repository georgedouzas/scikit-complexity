"""
# Steerable feedback loop

A receptor and an effector hold a product at its set point by negative feedback. A steering signal, such as a toxin
acting on the temperature receptors in fever, shifts the set point while it acts, and the loop returns once it ends.
"""

# %%
# The example builds `SteerableFeedbackLoopSystem` with a steering signal from time 20 to 40, and the observer saves
# every third frame with the default blueprint, whose primary view is the time series of the product and the effector.

import matplotlib.pyplot as plt
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import SteerableFeedbackLoopSystem

SYSTEM = 'steerable_feedback_loop'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=3)
trajectory = simulate(SteerableFeedbackLoopSystem(), observers=[observer])

# %%
# Set points and delay
# --------------------
# The set point solves `v / (1 + (x / K)^n) = d x`, with `K` raised by half during the signal. A long transmission
# delay should turn the regulation into oscillations around the set point.


def set_point(sensitivity):
    return brentq(lambda x: 2.0 / (1 + (x / sensitivity) ** 4) - x, 0.0, 3.0)


delayed = simulate(SteerableFeedbackLoopSystem(tau=3.0))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(trajectory.times, trajectory.get_values('loop.x'), label='delay 0.5')
ax.plot(delayed.times, delayed.get_values('loop.x'), alpha=0.6, label='delay 3')
ax.axhline(set_point(1.0), color='black', linestyle='--', label='set point')
ax.axhline(set_point(1.5), color='tab:red', linestyle='--', label='steered set point')
ax.axvspan(20, 40, color='gray', alpha=0.15, label='steering signal')
ax.set_xlabel('time')
ax.set_ylabel('product x')
ax.set_title('Steering shifts the set point, a long delay makes the loop oscillate')
ax.legend(fontsize=8)
plt.show()
