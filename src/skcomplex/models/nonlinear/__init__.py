"""Nonlinear dynamics: canonical chaotic flows and maps."""

from ._jerk import Jerk, sprott_jerk

__all__: list[str] = [
    'Jerk',
    'sprott_jerk',
]
