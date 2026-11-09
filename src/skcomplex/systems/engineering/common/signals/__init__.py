"""Signal realizations shared by the engineering packages.

They are the pulses that set states at given or periodic times.
"""

from ._pulse import PulseInteraction

__all__: list[str] = [
    'PulseInteraction',
]
