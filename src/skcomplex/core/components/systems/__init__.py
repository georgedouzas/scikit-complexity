"""Define systems, which compose elements and interactions on a space, in a time.

`CompositeSystem` composes components given as data, and `BaseSystem` is the base of a system class that builds its
components from model parameters.
"""

from ._base import BaseSystem, RunSettings
from ._composite import CompositeSystem

__all__: list[str] = [
    "BaseSystem",
    "CompositeSystem",
    "RunSettings",
]
