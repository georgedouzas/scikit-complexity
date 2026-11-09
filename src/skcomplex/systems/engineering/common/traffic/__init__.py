"""Traffic realizations shared by the engineering packages.

They are the adaptation of a speed or a flow towards a desired one.
"""

from ._speed_adaptation import SpeedAdaptationInteraction

__all__: list[str] = [
    'SpeedAdaptationInteraction',
]
