"""Mechanical realizations shared by the engineering packages.

They are the spring and damper between two coordinates.
"""

from ._spring_damper import SpringDamperInteraction

__all__: list[str] = [
    'SpringDamperInteraction',
]
