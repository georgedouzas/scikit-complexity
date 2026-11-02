"""Cosmology: the expansion of a homogeneous universe."""

from ._expansion import ScaleFactorElement, friedmann_expansion

__all__: list[str] = [
    'ScaleFactorElement',
    'friedmann_expansion',
]
