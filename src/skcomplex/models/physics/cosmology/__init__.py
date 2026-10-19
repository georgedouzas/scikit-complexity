"""Cosmology: the expansion of a homogeneous universe."""

from ._expansion import ScaleFactor, friedmann_expansion

__all__: list[str] = [
    'ScaleFactor',
    'friedmann_expansion',
]
