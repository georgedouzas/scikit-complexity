"""Control: feedback loops that keep or drive a system towards a goal."""

from ._control import pendulum_swing_up, pid_loop, satellite_transfer, thermostat

__all__: list[str] = [
    'pendulum_swing_up',
    'pid_loop',
    'satellite_transfer',
    'thermostat',
]
