"""Control: feedback loops that keep or drive a system towards a goal."""

from ._control import pendulum_swing_up, pid_loop, satellite_transfer, thermostat
from ._regulation import CartPendulumElement, DrivenOscillatorElement, lqr_oscillator, pid_inverted_pendulum

__all__: list[str] = [
    'CartPendulumElement',
    'DrivenOscillatorElement',
    'lqr_oscillator',
    'pendulum_swing_up',
    'pid_inverted_pendulum',
    'pid_loop',
    'satellite_transfer',
    'thermostat',
]
