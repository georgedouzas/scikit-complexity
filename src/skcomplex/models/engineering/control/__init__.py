"""Control: feedback loops that keep or drive a system towards a goal, and the controllers that close them."""

from ._control import pendulum_swing_up, pid_loop, satellite_transfer, thermostat
from ._integrator import IntegratorElement
from ._pid import PIDInteraction
from ._regulation import CartPendulumElement, DrivenOscillatorElement, lqr_oscillator, pid_inverted_pendulum
from ._state_feedback import StateFeedbackInteraction

__all__: list[str] = [
    'CartPendulumElement',
    'DrivenOscillatorElement',
    'IntegratorElement',
    'PIDInteraction',
    'StateFeedbackInteraction',
    'lqr_oscillator',
    'pendulum_swing_up',
    'pid_inverted_pendulum',
    'pid_loop',
    'satellite_transfer',
    'thermostat',
]
