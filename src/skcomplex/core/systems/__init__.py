"""Define systems, which are prepared once and simulated many times.

A system holds elements, interactions, nested systems, and a space, such as a ball and the law of its fall. `prepare`
turns its laws into a problem once, as `fit` does, and `simulate` runs it from a starting point, as `predict` does.
"""

from ._base import BaseSystem
from ._system import System

__all__: list[str] = [
    'BaseSystem',
    'System',
]
