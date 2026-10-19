"""Robotics: autonomous robots and robot swarms."""

from ._robots import LightSensitiveRobot, SwarmRobot, light_seeking_robots, swarmtaxis

__all__: list[str] = [
    'LightSensitiveRobot',
    'SwarmRobot',
    'light_seeking_robots',
    'swarmtaxis',
]
