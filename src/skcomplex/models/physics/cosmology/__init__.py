"""Cosmology: the expansion of a homogeneous universe."""

from ._expansion import ScaleFactorComponent, friedmann_expansion

__all__: list[str] = [
    'ScaleFactorComponent',
    'friedmann_expansion',
]
