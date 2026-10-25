"""Robotics: autonomous robots and robot swarms."""

from ._robots import LightSensitiveRobotComponent, SwarmRobotComponent, light_seeking_robots, swarmtaxis

__all__: list[str] = [
    'LightSensitiveRobotComponent',
    'SwarmRobotComponent',
    'light_seeking_robots',
    'swarmtaxis',
]
